# E3SM Microphysics Emulator Integration Guide

## Overview

This document provides a comprehensive guide for integrating the constraint-aware microphysics emulator into E3SM via FTorch. The emulator replaces computationally expensive cloud microphysics calculations with a neural network that maintains physical constraints.

### Model Architecture
- **Type**: Multi-head neural network with shared backbone
- **Input**: 11 features (cloud/rain properties, environmental variables)
- **Output**: 4 tendencies (qrtend, nctend, nrtend, qctend)
- **Constraints**: Mass conservation enforced (`qctend = -qrtend`)

---

## Performance Benchmarks

Benchmarked on **500 samples × 100 iterations** (50,000 total inferences) on CPU:

| Metric | Option 1: Inference Only | Option 2: With Preprocessing |
|--------|--------------------------|------------------------------|
| **Total Time** | 22.10 s | 22.06 s |
| **Avg per Sample** | 442.1 µs | 441.2 µs |
| Preprocessing | — | 0.19 µs (0.04%) |
| Inference | 442.1 µs | 440.8 µs (99.9%) |
| Postprocessing | — | 0.14 µs (0.03%) |

### Key Finding
**Preprocessing/postprocessing overhead is negligible** (~0.33 µs or 0.07% of total time). 

**Recommendation**: Embed preprocessing in Fortran (Option 2) for seamless E3SM integration. This eliminates external dependencies and simplifies the data pipeline.

---

## Files to Provide

### Core Model Files
| File | Description |
|------|-------------|
| `emulator_for_e3sm.pt` | TorchScript model (the neural network) |
| `input_scaler_params.txt` | Input normalization parameters (mean, scale) |
| `output_scaler_params.txt` | Output normalization parameters (mean, scale) |

### Reference Fortran Code
| File | Description |
|------|-------------|
| `sample_ftorch2.f90` | Minimal inference example |
| `benchmark_with_preprocessing.f90` | Full pipeline with preprocessing |
| `validate_emulator.f90` | Validation against Python outputs |

### Test Data (for validation)
| File | Description |
|------|-------------|
| `physical_test_inputs.txt` | Test inputs in physical units |
| `physical_test_outputs.txt` | Expected outputs in physical units |
| `normalized_test_inputs.txt` | Test inputs (normalized) |
| `normalized_test_outputs.txt` | Expected outputs (normalized) |

---

## Data Preprocessing Pipeline

### Input Features (11 total)

| Index | Name | Log Transform? | Description |
|-------|------|----------------|-------------|
| 0 | QC_TAU_in | ✓ | Cloud water mixing ratio |
| 1 | QR_TAU_in | ✓ | Rain water mixing ratio |
| 2 | NC_TAU_in | ✓ | Cloud droplet number |
| 3 | NR_TAU_in | ✓ | Rain droplet number |
| 4 | PGAM | ✗ | Gamma distribution parameter |
| 5 | LAMC | ✓ | Cloud lambda parameter |
| 6 | LAMR | ✓ | Rain lambda parameter |
| 7 | N0R | ✓ | Rain intercept parameter |
| 8 | RHO_CLUBB | ✗ | Air density |
| 9 | CLOUD | ✗ | Cloud fraction |
| 10 | FREQR | ✗ | Rain frequency |

### Output Tendencies (4 total)

| Index | Name | Description | Physical Constraint |
|-------|------|-------------|---------------------|
| 0 | qrtend | Rain tendency | — |
| 1 | nctend | Cloud number tendency | — |
| 2 | nrtend | Rain number tendency | — |
| 3 | qctend | Cloud water tendency | `= -qrtend` (mass conservation) |

### Preprocessing Steps (Input)

```fortran
! 1. Log transform specific columns (indices 1,2,3,4,6,7,8 in 1-indexed Fortran)
do k = 1, NUM_LOG_INPUTS
    j = LOG_INPUT_INDICES(k)  ! = (/1, 2, 3, 4, 6, 7, 8/)
    features(j) = log10(features(j) + 1.0e-10)
end do

! 2. StandardScaler normalization
do j = 1, NUM_FEATURES
    features(j) = (features(j) - input_mean(j)) / input_scale(j)
end do
```

### Postprocessing Steps (Output)

```fortran
! 1. Inverse StandardScaler
do j = 1, NUM_OUTPUTS
    tendencies(j) = tendencies(j) * output_scale(j) + output_mean(j)
end do

! 2. Inverse log transform (sign-preserving)
do j = 1, NUM_OUTPUTS
    if (tendencies(j) >= 0.0) then
        physical_output(j) = (10.0 ** tendencies(j)) - 1.0e-10
    else
        physical_output(j) = -((10.0 ** abs(tendencies(j))) - 1.0e-10)
    end if
end do
```

---

## Integration Steps for E3SM Collaborator

### Prerequisites
1. **FTorch** library compiled and installed
2. **libtorch** (PyTorch C++ libraries)
3. Fortran compiler (gfortran, ifort, or ftn)

### Step 1: Add Files to E3SM

Place these files in the appropriate E3SM directory:
```
e3sm/components/eam/src/physics/cam/
├── emulator_for_e3sm.pt          # TorchScript model
├── input_scaler_params.txt       # Input normalization
├── output_scaler_params.txt      # Output normalization
└── microphysics_emulator.F90     # New Fortran module (see below)
```

### Step 2: Create Fortran Module

Create `microphysics_emulator.F90` based on `benchmark_with_preprocessing.f90`:

```fortran
module microphysics_emulator
    use iso_fortran_env, only: real32
    use ftorch
    implicit none
    
    private
    public :: init_emulator, run_emulator, cleanup_emulator
    
    ! Module variables
    type(torch_model), save :: model
    real(real32), dimension(11), save :: input_mean, input_scale
    real(real32), dimension(4), save :: output_mean, output_scale
    logical, save :: initialized = .false.
    
contains
    
    subroutine init_emulator(model_path, scaler_dir)
        ! Load model and scaler parameters once at initialization
        character(len=*), intent(in) :: model_path, scaler_dir
        ! ... (load model and scalers)
        initialized = .true.
    end subroutine
    
    subroutine run_emulator(physical_inputs, physical_outputs, n_points)
        ! Main entry point: takes physical inputs, returns physical outputs
        real(real32), intent(in) :: physical_inputs(:,:)
        real(real32), intent(out) :: physical_outputs(:,:)
        integer, intent(in) :: n_points
        ! ... (preprocess -> inference -> postprocess)
    end subroutine
    
    subroutine cleanup_emulator()
        call torch_delete(model)
        initialized = .false.
    end subroutine
    
end module
```

### Step 3: Modify E3SM Build System

Add to `CMakeLists.txt` or build configuration:
```cmake
# Link FTorch and libtorch
target_link_libraries(eam 
    ftorch
    torch torch_cpu c10
)

# Include FTorch modules
target_include_directories(eam PRIVATE ${FTORCH_DIR}/modules)
```

### Step 4: Call Emulator from Microphysics

In the relevant microphysics routine:
```fortran
use microphysics_emulator, only: run_emulator

! Replace expensive calculation with emulator
call run_emulator(input_features, output_tendencies, ncol)
```

---

## Validation Checklist

Before deploying, verify:

- [ ] Output matches Python reference within tolerance (~10⁻⁶)
- [ ] Mass conservation: `|qctend + qrtend| < 10⁻⁶`
- [ ] Physical bounds respected (check for NaN/Inf)
- [ ] Performance acceptable for target resolution

### Run Validation
```bash
./validate_emulator emulator_for_e3sm.pt /path/to/test_data/
```

Expected output: All samples PASSED with max error ~10⁻⁷

---

## Quick Reference

### Compilation Command
```bash
gfortran -O3 your_code.f90 -o your_program \
    -I$FTORCH_BUILD/modules -L$FTORCH_BUILD -lftorch \
    -L$TORCH_LIB_PATH -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,$FTORCH_BUILD -Wl,-rpath,$TORCH_LIB_PATH
```

### Minimal Inference Example
```fortran
use ftorch
type(torch_model) :: model
real(real32), target :: input(1,11), output(1,4)

call torch_model_load(model, 'emulator_for_e3sm.pt', torch_kCPU)
! ... set input values (normalized!) ...
call torch_model_forward(model, input_tensors, output_tensors)
! ... output contains normalized tendencies ...
call torch_delete(model)
```

