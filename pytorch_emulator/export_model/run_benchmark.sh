#!/bin/bash
#
# Benchmark comparison: Option 1 (inference only) vs Option 2 (with preprocessing)
#

set -e

# Change to script directory
cd "$(dirname "$0")"

echo "============================================================"
echo "Setting up benchmark environment"
echo "============================================================"

# Check if conda environment variables are set
if [ -z "$FTORCH_BUILD" ]; then
    echo "WARNING: FTORCH_BUILD not set. Trying to source conda..."
    source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
    conda activate rocky9-pytorch
fi

# Step 1: Export scaler parameters and create physical test data
echo ""
echo "Step 1: Exporting scaler parameters..."
python export_scalers_for_fortran.py

# Step 2: Compile the benchmark programs
echo ""
echo "Step 2: Compiling benchmark programs..."

COMPILE_FLAGS="-O3 -march=native"
FTORCH_FLAGS="-I$FTORCH_BUILD/modules -L$FTORCH_BUILD -lftorch"
TORCH_FLAGS="-L$TORCH_LIB_PATH -L$CONDA_LIB_PATH -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lstdc++"
RPATH_FLAGS="-Wl,-rpath,$FTORCH_BUILD -Wl,-rpath,$TORCH_LIB_PATH -Wl,-rpath,$CONDA_LIB_PATH -Wl,-rpath-link,$TORCH_LIB_PATH -Wl,-rpath-link,$CONDA_LIB_PATH"

echo "Compiling benchmark_without_preprocessing..."
gfortran $COMPILE_FLAGS benchmark_without_preprocessing.f90 -o benchmark_without_preprocessing \
    $FTORCH_FLAGS $TORCH_FLAGS $RPATH_FLAGS

echo "Compiling benchmark_with_preprocessing..."
gfortran $COMPILE_FLAGS benchmark_with_preprocessing.f90 -o benchmark_with_preprocessing \
    $FTORCH_FLAGS $TORCH_FLAGS $RPATH_FLAGS

echo "Compilation complete!"

# Step 3: Run benchmarks
echo ""
echo "============================================================"
echo "Running benchmarks..."
echo "============================================================"

echo ""
echo "--- OPTION 1: Inference Only (preprocessing external) ---"
./benchmark_without_preprocessing emulator_for_e3sm.pt .

echo ""
echo ""
echo "--- OPTION 2: With Preprocessing in Fortran ---"
./benchmark_with_preprocessing emulator_for_e3sm.pt .

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "============================================================"


