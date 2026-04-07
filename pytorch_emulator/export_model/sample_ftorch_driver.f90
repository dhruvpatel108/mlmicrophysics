program run_emulator
  use iso_fortran_env, only: int64, real64
  use ftorch
  implicit none

  type(torch_model) :: emulator
  type(torch_tensor) :: input_tensor, output_tensor
  real(real64), dimension(11) :: features
  real(real64), dimension(4) :: tendencies
  integer :: ierr
  integer(int64), dimension(2) :: input_shape
  integer(int64), dimension(2) :: output_shape

  call torch_init(ierr)
  call check_status('torch_init', ierr)

  call torch_model_load(emulator, 'emulator_for_e3sm.pt', ierr)
  call check_status('torch_model_load', ierr)

  features = (/ 0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
                0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
                0.0_real64 /)
  input_shape = [1_int64, 11_int64]

  call torch_tensor_from_array(input_tensor, features, input_shape, ierr)
  call check_status('torch_tensor_from_array', ierr)

  call torch_model_forward(emulator, input_tensor, output_tensor, ierr)
  call check_status('torch_model_forward', ierr)

  output_shape = [1_int64, 4_int64]
  call torch_tensor_to_array(output_tensor, tendencies, output_shape, ierr)
  call check_status('torch_tensor_to_array', ierr)

  write(*,'(A)') 'Emulator tendencies (qrtend, nctend, nrtend, qctend):'
  write(*,'(4(1X,ES12.5))') tendencies

  call torch_tensor_delete(input_tensor, ierr)
  call torch_tensor_delete(output_tensor, ierr)
  call torch_model_delete(emulator, ierr)
  call torch_finalize(ierr)
contains
  subroutine check_status(label, ierr)
    character(len=*), intent(in) :: label
    integer, intent(in) :: ierr
    if (ierr /= 0) then
      write(*,'(A,1X,I0)') trim(label)//' failed with code', ierr
      stop 1
    end if
  end subroutine check_status
end program run_emulator
