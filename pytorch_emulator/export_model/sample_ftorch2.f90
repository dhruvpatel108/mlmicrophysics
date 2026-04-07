program run_emulator
  use iso_fortran_env, only : real32
  use ftorch, only : torch_model, torch_tensor, torch_tensor_from_array, &
                     torch_model_load, torch_model_forward, torch_delete, &
                     torch_kCPU
  implicit none

  type(torch_model) :: emulator
  type(torch_tensor), dimension(1) :: input_tensors
  type(torch_tensor), dimension(1) :: output_tensors

  ! Fortran arrays that back the Torch tensors (must have TARGET)
  real(real32), target, dimension(1, 11) :: features
  real(real32), target, dimension(1, 4)  :: tendencies  ! 4 outputs: qrtend, nctend, nrtend, qctend

  ! Load trained model onto CPU
  call torch_model_load(emulator, 'emulator_for_e3sm.pt', torch_kCPU)

  ! Prepare host-side buffers and expose them as Torch tensors
  ! features = 0.5_real32
  features(1,:) = (/ 9.017912745475769e-01_real32, -1.506153047084808e-01_real32, &
                  -1.628314554691315e-01_real32, -8.916091918945312e-01_real32, &
                    5.380609631538391e-01_real32, -1.152362823486328e+00_real32, &
                  -2.476505935192108e-01_real32, -1.269024014472961e+00_real32, &
                  -3.144663870334625e-01_real32, -1.678047657012939e+00_real32, &
                  -7.403166294097900e-01_real32 /)
                  
  tendencies = 0.0_real32
  call torch_tensor_from_array(input_tensors(1), features, torch_kCPU)
  call torch_tensor_from_array(output_tensors(1), tendencies, torch_kCPU)

  ! Forward pass (writes predictions directly into tendencies array)
  call torch_model_forward(emulator, input_tensors, output_tensors)

  write(*,*) 'Emulator output:'
  write(*,*) tendencies

  ! Cleanup Torch objects (Fortran arrays remain owned by user code)
  call torch_delete(output_tensors)
  call torch_delete(input_tensors)
  call torch_delete(emulator)

end program run_emulator