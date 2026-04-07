program test_standalone_emulator
  !
  ! Test the standalone emulator with embedded preprocessing/postprocessing.
  !
  ! This model takes PHYSICAL inputs and returns PHYSICAL outputs.
  ! No external preprocessing or scaler files needed!
  !
  use iso_fortran_env, only : real32
  use ftorch, only : torch_model, torch_tensor, torch_tensor_from_array, &
                     torch_model_load, torch_model_forward, torch_delete, &
                     torch_kCPU
  implicit none

  type(torch_model) :: emulator
  type(torch_tensor), dimension(1) :: input_tensors
  type(torch_tensor), dimension(1) :: output_tensors

  ! Input: 11 PHYSICAL features (no preprocessing needed!)
  ! Output: 4 PHYSICAL tendencies (no postprocessing needed!)
  real(real32), target, dimension(1, 11) :: physical_input
  real(real32), target, dimension(1, 4)  :: physical_output

  ! Load the standalone model
  write(*,*) '============================================'
  write(*,*) 'Testing Standalone Emulator'
  write(*,*) '(with embedded preprocessing/postprocessing)'
  write(*,*) '============================================'
  write(*,*) ''
  
  call torch_model_load(emulator, 'emulator_standalone.pt', torch_kCPU)
  write(*,*) 'Model loaded successfully!'
  write(*,*) ''

  ! Set PHYSICAL input values (typical E3SM values)
  ! These are raw physical values - no log transform or scaling needed!
  physical_input(1, 1) = 1.0e-5_real32    ! QC_TAU_in (cloud water mixing ratio)
  physical_input(1, 2) = 1.0e-6_real32    ! QR_TAU_in (rain water mixing ratio)
  physical_input(1, 3) = 1.0e8_real32     ! NC_TAU_in (cloud droplet number)
  physical_input(1, 4) = 1.0e5_real32     ! NR_TAU_in (rain droplet number)
  physical_input(1, 5) = 10.0_real32      ! PGAM
  physical_input(1, 6) = 1.0e5_real32     ! LAMC
  physical_input(1, 7) = 1.0e3_real32     ! LAMR
  physical_input(1, 8) = 1.0e6_real32     ! N0R
  physical_input(1, 9) = 0.8_real32       ! RHO_CLUBB
  physical_input(1, 10) = 0.5_real32      ! CLOUD
  physical_input(1, 11) = 0.3_real32      ! FREQR
  
  physical_output = 0.0_real32

  write(*,*) 'Physical inputs:'
  write(*,'(A,E12.4)') '  QC_TAU_in  = ', physical_input(1, 1)
  write(*,'(A,E12.4)') '  QR_TAU_in  = ', physical_input(1, 2)
  write(*,'(A,E12.4)') '  NC_TAU_in  = ', physical_input(1, 3)
  write(*,'(A,E12.4)') '  NR_TAU_in  = ', physical_input(1, 4)
  write(*,'(A,E12.4)') '  PGAM       = ', physical_input(1, 5)
  write(*,'(A,E12.4)') '  LAMC       = ', physical_input(1, 6)
  write(*,'(A,E12.4)') '  LAMR       = ', physical_input(1, 7)
  write(*,'(A,E12.4)') '  N0R        = ', physical_input(1, 8)
  write(*,'(A,E12.4)') '  RHO_CLUBB  = ', physical_input(1, 9)
  write(*,'(A,E12.4)') '  CLOUD      = ', physical_input(1, 10)
  write(*,'(A,E12.4)') '  FREQR      = ', physical_input(1, 11)
  write(*,*) ''

  ! Create tensors and run inference
  call torch_tensor_from_array(input_tensors(1), physical_input, torch_kCPU)
  call torch_tensor_from_array(output_tensors(1), physical_output, torch_kCPU)
  call torch_model_forward(emulator, input_tensors, output_tensors)

  ! Output is directly in PHYSICAL units - no postprocessing needed!
  write(*,*) 'Physical outputs (tendencies):'
  write(*,'(A,E15.8)') '  qrtend = ', physical_output(1, 1)
  write(*,'(A,E15.8)') '  nctend = ', physical_output(1, 2)
  write(*,'(A,E15.8)') '  nrtend = ', physical_output(1, 3)
  write(*,'(A,E15.8)') '  qctend = ', physical_output(1, 4)
  write(*,*) ''
  
  ! Verify mass conservation (qctend should equal -qrtend)
  write(*,*) 'Mass conservation check:'
  write(*,'(A,E15.8)') '  qctend + qrtend = ', physical_output(1, 4) + physical_output(1, 1)
  write(*,*) ''

  ! Cleanup
  call torch_delete(output_tensors)
  call torch_delete(input_tensors)
  call torch_delete(emulator)
  
  write(*,*) '============================================'
  write(*,*) 'Test completed successfully!'
  write(*,*) '============================================'

end program test_standalone_emulator

