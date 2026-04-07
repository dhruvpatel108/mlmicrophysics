program benchmark_with_preprocessing
  !
  ! OPTION 2: Full preprocessing/postprocessing embedded in Fortran
  !
  ! This benchmark measures the time for:
  ! 1. Load physical (raw) input data
  ! 2. Preprocess: log transform + standardization
  ! 3. Run model inference
  ! 4. Postprocess: inverse standardization + inverse log transform
  !
  use iso_fortran_env, only : real32, real64, int64
  use ftorch, only : torch_model, torch_tensor, torch_tensor_from_array, &
                     torch_model_load, torch_model_forward, torch_delete, &
                     torch_kCPU
  implicit none

  ! Configuration
  integer, parameter :: NUM_FEATURES = 11
  integer, parameter :: NUM_OUTPUTS = 4
  integer, parameter :: NUM_ITERATIONS = 100  ! Number of inference iterations for timing
  
  ! Which input features get log-transformed (1-indexed)
  integer, parameter :: NUM_LOG_INPUTS = 7
  integer, dimension(NUM_LOG_INPUTS), parameter :: LOG_INPUT_INDICES = (/1, 2, 3, 4, 6, 7, 8/)
  real(real32), parameter :: LOG_EPSILON = 1.0e-10
  
  ! FTorch objects
  type(torch_model) :: emulator
  type(torch_tensor), dimension(1) :: input_tensors
  type(torch_tensor), dimension(1) :: output_tensors
  
  ! Data arrays
  real(real32), allocatable, target :: physical_inputs(:,:)
  real(real32), allocatable, target :: features(:,:)
  real(real32), allocatable, target :: tendencies(:,:)
  real(real32), allocatable :: physical_outputs(:,:)
  real(real32), allocatable :: expected_outputs(:,:)
  
  ! Scaler parameters
  real(real32), dimension(NUM_FEATURES) :: input_mean, input_scale
  real(real32), dimension(NUM_OUTPUTS) :: output_mean, output_scale
  
  ! Working variables
  integer :: num_samples, i, j, k, iter, ios, unit_in
  character(len=512) :: line
  character(len=256) :: model_path, data_dir
  
  ! Timing variables
  integer(int64) :: count_start, count_end, count_rate
  real(real64) :: time_preprocess, time_inference, time_postprocess, time_total
  real(real64) :: avg_preprocess, avg_inference, avg_postprocess, avg_total
  
  ! Default paths
  model_path = 'emulator_for_e3sm.pt'
  data_dir = '.'
  
  ! Parse command line
  if (command_argument_count() >= 1) call get_command_argument(1, model_path)
  if (command_argument_count() >= 2) call get_command_argument(2, data_dir)
  
  write(*,*) '============================================================'
  write(*,*) 'OPTION 2: Benchmark WITH Preprocessing in Fortran'
  write(*,*) '============================================================'
  write(*,*) 'Model: ', trim(model_path)
  write(*,*) 'Data dir: ', trim(data_dir)
  write(*,'(A,I0)') ' Iterations: ', NUM_ITERATIONS
  write(*,*) ''
  
  ! Load scaler parameters
  call load_scaler_params(trim(data_dir)//'/input_scaler_params.txt', input_mean, input_scale, NUM_FEATURES)
  call load_scaler_params(trim(data_dir)//'/output_scaler_params.txt', output_mean, output_scale, NUM_OUTPUTS)
  write(*,*) 'Loaded scaler parameters'
  
  ! Load physical test inputs
  call load_test_data(trim(data_dir)//'/physical_test_inputs.txt', physical_inputs, num_samples, NUM_FEATURES)
  write(*,'(A,I0,A)') ' Loaded ', num_samples, ' physical test samples'
  
  ! Load expected physical outputs for verification
  call load_test_data(trim(data_dir)//'/physical_test_outputs.txt', expected_outputs, num_samples, NUM_OUTPUTS)
  
  ! Allocate working arrays
  allocate(features(1, NUM_FEATURES))
  allocate(tendencies(1, NUM_OUTPUTS))
  allocate(physical_outputs(num_samples, NUM_OUTPUTS))
  
  ! Load TorchScript model
  call torch_model_load(emulator, trim(model_path), torch_kCPU)
  write(*,*) 'Model loaded'
  write(*,*) ''
  
  ! Initialize timing
  call system_clock(count_rate=count_rate)
  time_preprocess = 0.0d0
  time_inference = 0.0d0
  time_postprocess = 0.0d0
  
  write(*,'(A,I0,A)') ' Running ', NUM_ITERATIONS, ' iterations...'
  
  ! Main timing loop
  do iter = 1, NUM_ITERATIONS
    do i = 1, num_samples
      ! ============ PREPROCESSING ============
      call system_clock(count_start)
      
      ! Copy physical input
      features(1,:) = physical_inputs(i,:)
      
      ! Log transform specific columns
      do k = 1, NUM_LOG_INPUTS
        j = LOG_INPUT_INDICES(k)
        features(1,j) = log10(features(1,j) + LOG_EPSILON)
      end do
      
      ! StandardScaler: (x - mean) / scale
      do j = 1, NUM_FEATURES
        features(1,j) = (features(1,j) - input_mean(j)) / input_scale(j)
      end do
      
      call system_clock(count_end)
      time_preprocess = time_preprocess + dble(count_end - count_start) / dble(count_rate)
      
      ! ============ INFERENCE ============
      call system_clock(count_start)
      
      tendencies = 0.0_real32
      call torch_tensor_from_array(input_tensors(1), features, torch_kCPU)
      call torch_tensor_from_array(output_tensors(1), tendencies, torch_kCPU)
      call torch_model_forward(emulator, input_tensors, output_tensors)
      
      call system_clock(count_end)
      time_inference = time_inference + dble(count_end - count_start) / dble(count_rate)
      
      ! ============ POSTPROCESSING ============
      call system_clock(count_start)
      
      ! Inverse StandardScaler: x * scale + mean
      do j = 1, NUM_OUTPUTS
        tendencies(1,j) = tendencies(1,j) * output_scale(j) + output_mean(j)
      end do
      
      ! Inverse log transform (sign-preserving)
      do j = 1, NUM_OUTPUTS
        if (tendencies(1,j) >= 0.0_real32) then
          physical_outputs(i,j) = (10.0_real32 ** tendencies(1,j)) - LOG_EPSILON
        else
          physical_outputs(i,j) = -((10.0_real32 ** abs(tendencies(1,j))) - LOG_EPSILON)
        end if
      end do
      
      call system_clock(count_end)
      time_postprocess = time_postprocess + dble(count_end - count_start) / dble(count_rate)
      
      ! Cleanup tensors
      call torch_delete(output_tensors)
      call torch_delete(input_tensors)
    end do
  end do
  
  ! Calculate averages
  time_total = time_preprocess + time_inference + time_postprocess
  avg_preprocess = time_preprocess / (NUM_ITERATIONS * num_samples)
  avg_inference = time_inference / (NUM_ITERATIONS * num_samples)
  avg_postprocess = time_postprocess / (NUM_ITERATIONS * num_samples)
  avg_total = time_total / (NUM_ITERATIONS * num_samples)
  
  ! Print timing results
  write(*,*) ''
  write(*,*) '============================================================'
  write(*,*) 'TIMING RESULTS (Option 2: With Preprocessing)'
  write(*,*) '============================================================'
  write(*,'(A,F12.6,A)') ' Total time:        ', time_total, ' seconds'
  write(*,'(A,F12.6,A)') '   Preprocessing:   ', time_preprocess, ' seconds'
  write(*,'(A,F12.6,A)') '   Inference:       ', time_inference, ' seconds'
  write(*,'(A,F12.6,A)') '   Postprocessing:  ', time_postprocess, ' seconds'
  write(*,*) ''
  write(*,'(A,E12.4,A)') ' Avg per sample:    ', avg_total * 1.0d6, ' microseconds'
  write(*,'(A,E12.4,A)') '   Preprocessing:   ', avg_preprocess * 1.0d6, ' microseconds'
  write(*,'(A,E12.4,A)') '   Inference:       ', avg_inference * 1.0d6, ' microseconds'
  write(*,'(A,E12.4,A)') '   Postprocessing:  ', avg_postprocess * 1.0d6, ' microseconds'
  write(*,*) ''
  
  ! Verify outputs (last iteration)
  write(*,*) '============================================================'
  write(*,*) 'OUTPUT VERIFICATION (last iteration, sample 1)'
  write(*,*) '============================================================'
  write(*,*) 'Computed vs Expected physical outputs:'
  do j = 1, NUM_OUTPUTS
    write(*,'(A,I1,A,E15.8,A,E15.8)') '  output[', j-1, ']: ', &
      physical_outputs(1,j), ' vs ', expected_outputs(1,j)
  end do
  
  ! Cleanup
  call torch_delete(emulator)
  deallocate(physical_inputs, features, tendencies, physical_outputs, expected_outputs)

contains

  subroutine load_scaler_params(filename, means, scales, n)
    character(len=*), intent(in) :: filename
    real(real32), intent(out) :: means(:), scales(:)
    integer, intent(in) :: n
    integer :: unit_num, i, ios
    character(len=512) :: line
    
    open(newunit=unit_num, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: Could not open ', trim(filename)
      stop 1
    end if
    
    ! Skip comment lines
    do
      read(unit_num, '(A)') line
      if (line(1:1) /= '#') exit
    end do
    backspace(unit_num)
    
    ! Skip remaining comment lines and read data
    do
      read(unit_num, '(A)', iostat=ios) line
      if (ios /= 0) exit
      if (line(1:1) == '#') cycle
      backspace(unit_num)
      exit
    end do
    
    do i = 1, n
      read(unit_num, *, iostat=ios) means(i), scales(i)
      if (ios /= 0) then
        write(*,*) 'ERROR reading scaler params at line ', i
        stop 1
      end if
    end do
    close(unit_num)
  end subroutine

  subroutine load_test_data(filename, data, n_samples, n_cols)
    character(len=*), intent(in) :: filename
    real(real32), allocatable, intent(out) :: data(:,:)
    integer, intent(out) :: n_samples
    integer, intent(in) :: n_cols
    integer :: unit_num, i, j, ios, n_cols_file
    character(len=512) :: line
    
    open(newunit=unit_num, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: Could not open ', trim(filename)
      stop 1
    end if
    
    read(unit_num, '(A)') line
    read(line(2:), *) n_samples, n_cols_file
    
    allocate(data(n_samples, n_cols))
    do i = 1, n_samples
      read(unit_num, *) (data(i,j), j=1, n_cols)
    end do
    close(unit_num)
  end subroutine

end program benchmark_with_preprocessing


