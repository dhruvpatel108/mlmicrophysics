#!/bin/bash
#SBATCH --account=m4942
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --job-name=test_data_loaders
#SBATCH --output=test_data_loaders-%j.out
#SBATCH --error=test_data_loaders-%j.err

# Load environment
module load conda
conda activate mlmicrophysics-env

# Change to project directory
cd /global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/pytorch_emulator

# Set Python path
export PYTHONPATH="/global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/pytorch_emulator:$PYTHONPATH"

echo "🚀 Testing Data Loader Performance"
echo "=================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo ""


echo ""
echo "🧪 Testing CURRENT STREAMING Data Loader"
echo "========================================"
python scripts/test_data_loaders.py \
    --config configs/multi_gpu_test.yml \
    --loader current \
    --max-batches 3000 \
    --max-time 500

# Test optimized streaming loader first (most likely to work)
echo "🧪 Testing OPTIMIZED STREAMING Data Loader"
echo "============================================"
python scripts/test_data_loaders.py \
    --config configs/optimized_streaming_test.yml \
    --loader optimized \
    --max-batches 3000 \
    --max-time 500

echo ""
echo "🧪 Testing DASK Data Loader"
echo "==========================="
python scripts/test_data_loaders.py \
    --config configs/dask_test.yml \
    --loader dask \
    --max-batches 3000 \
    --max-time 500


echo ""
echo "🏁 All tests completed!"
echo "Check the output files for performance results." 