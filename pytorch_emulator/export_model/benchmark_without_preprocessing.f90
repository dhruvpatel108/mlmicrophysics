program benchmark_without_preprocessing
  !
  ! OPTION 1: Only model inference (preprocessing done externally)
  !
  ! This benchmark measures the time for ONLY model inference
  ! Input data is already normalized (preprocessing done in Python)
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
  
  ! FTorch objects
  type(torch_model) :: emulator
  type(torch_tensor), dimension(1) :: input_tensors
  type(torch_tensor), dimension(1) :: output_tensors
  
  ! Data arrays
  real(real32), allocatable, target :: normalized_inputs(:,:)
  real(real32), allocatable, target :: features(:,:)
  real(real32), allocatable, target :: tendencies(:,:)
  real(real32), allocatable :: normalized_outputs(:,:)
  real(real32), allocatable :: expected_outputs(:,:)
  
  ! Working variables
  integer :: num_samples, i, j, iter, ios
  character(len=512) :: line
  character(len=256) :: model_path, data_dir
  
  ! Timing variables
  integer(int64) :: count_start, count_end, count_rate
  real(real64) :: time_inference, time_total
  real(real64) :: avg_inference
  
  ! Default paths
  model_path = 'emulator_for_e3sm.pt'
  data_dir = '.'
  
  ! Parse command line
  if (command_argument_count() >= 1) call get_command_argument(1, model_path)
  if (command_argument_count() >= 2) call get_command_argument(2, data_dir)
  
  write(*,*) '============================================================'
  write(*,*) 'OPTION 1: Benchmark WITHOUT Preprocessing (inference only)'
  write(*,*) '============================================================'
  write(*,*) 'Model: ', trim(model_path)
  write(*,*) 'Data dir: ', trim(data_dir)
  write(*,'(A,I0)') ' Iterations: ', NUM_ITERATIONS
  write(*,*) ''
  
  ! Load normalized test inputs (already preprocessed)
  call load_test_data(trim(data_dir)//'/normalized_test_inputs.txt', normalized_inputs, num_samples, NUM_FEATURES)
  write(*,'(A,I0,A)') ' Loaded ', num_samples, ' normalized test samples'
  
  ! Load expected normalized outputs for verification
  call load_test_data(trim(data_dir)//'/normalized_test_outputs.txt', expected_outputs, num_samples, NUM_OUTPUTS)
  
  ! Allocate working arrays
  allocate(features(1, NUM_FEATURES))
  allocate(tendencies(1, NUM_OUTPUTS))
  allocate(normalized_outputs(num_samples, NUM_OUTPUTS))
  
  ! Load TorchScript model
  call torch_model_load(emulator, trim(model_path), torch_kCPU)
  write(*,*) 'Model loaded'
  write(*,*) ''
  
  ! Initialize timing
  call system_clock(count_rate=count_rate)
  time_inference = 0.0d0
  
  write(*,'(A,I0,A)') ' Running ', NUM_ITERATIONS, ' iterations...'
  
  ! Main timing loop
  do iter = 1, NUM_ITERATIONS
    do i = 1, num_samples
      ! ============ INFERENCE ONLY ============
      call system_clock(count_start)
      
      ! Copy normalized input directly
      features(1,:) = normalized_inputs(i,:)
      tendencies = 0.0_real32
      
      call torch_tensor_from_array(input_tensors(1), features, torch_kCPU)
      call torch_tensor_from_array(output_tensors(1), tendencies, torch_kCPU)
      call torch_model_forward(emulator, input_tensors, output_tensors)
      
      ! Store output (already in normalized space)
      normalized_outputs(i,:) = tendencies(1,:)
      
      call system_clock(count_end)
      time_inference = time_inference + dble(count_end - count_start) / dble(count_rate)
      
      ! Cleanup tensors
      call torch_delete(output_tensors)
      call torch_delete(input_tensors)
    end do
  end do
  
  ! Calculate averages
  time_total = time_inference
  avg_inference = time_inference / (NUM_ITERATIONS * num_samples)
  
  ! Print timing results
  write(*,*) ''
  write(*,*) '============================================================'
  write(*,*) 'TIMING RESULTS (Option 1: Inference Only)'
  write(*,*) '============================================================'
  write(*,'(A,F12.6,A)') ' Total time:        ', time_total, ' seconds'
  write(*,'(A,F12.6,A)') '   Inference:       ', time_inference, ' seconds'
  write(*,*) ''
  write(*,'(A,E12.4,A)') ' Avg per sample:    ', avg_inference * 1.0d6, ' microseconds'
  write(*,*) ''
  
  ! Verify outputs (last iteration)
  write(*,*) '============================================================'
  write(*,*) 'OUTPUT VERIFICATION (last iteration, sample 1)'
  write(*,*) '============================================================'
  write(*,*) 'Computed vs Expected normalized outputs:'
  do j = 1, NUM_OUTPUTS
    write(*,'(A,I1,A,E15.8,A,E15.8)') '  output[', j-1, ']: ', &
      normalized_outputs(1,j), ' vs ', expected_outputs(1,j)
  end do
  
  ! Cleanup
  call torch_delete(emulator)
  deallocate(normalized_inputs, features, tendencies, normalized_outputs, expected_outputs)

contains

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

end program benchmark_without_preprocessing


