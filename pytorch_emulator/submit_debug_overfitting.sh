#!/bin/bash
#SBATCH --job-name=debug_overfitting
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=debug                    # Use debug queue for quick testing
#SBATCH --time=0:30:00                 # 30 minute time limit
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks-per-node=1            # Single task
#SBATCH --cpus-per-task=8              # 8 CPUs for single GPU
#SBATCH --gpus-per-node=1              # 1 A100 GPU for debug
#SBATCH --mem=32G                      # 32GB memory
#SBATCH --output=debug_overfitting-%j.out
#SBATCH --error=debug_overfitting-%j.err

# Print job info
echo "🧪 DEBUG OVERFITTING TEST"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "==========================================="

# Load environment
echo "🔧 Loading conda environment..."
module load conda
conda activate mlmicrophysics-env

# Verify GPU access
echo "🔍 Checking GPU availability..."
nvidia-smi

# Change to project directory
cd /global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/pytorch_emulator

# Run debug overfitting test
echo "🚀 Starting DEBUG OVERFITTING TEST..."
echo "📊 Using 50 parquet files, no subsampling, 20 epochs"
echo "🎯 Goal: Check if model can overfit on small dataset"

python scripts/train_streaming_parallel.py \
    --config configs/debug_overfitting_test.yml \
    --multi_gpu \
    --verbose

echo "✅ DEBUG OVERFITTING TEST COMPLETED!"
echo "End time: $(date)" 