#!/bin/bash
#SBATCH --job-name=optimized_production_full
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=regular                  # Use regular queue for production
#SBATCH --time=24:00:00                # 21 hour time limit for full training
#SBATCH --nodes=1                      # Single node for stability
#SBATCH --ntasks-per-node=1            # Single task per node for multi-GPU
#SBATCH --cpus-per-task=16             # 16 CPUs for multi-GPU
#SBATCH --gpus-per-node=4              # 4 A100 GPUs (single node)
#SBATCH --mem=64G                     # 64GB memory
#SBATCH --output=optimized_production_fixed_architecture-%j.out
#SBATCH --error=optimized_production_fixed_architecture-%j.err

# Print job info
echo "🚀 OPTIMIZED STREAMING FULL PRODUCTION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: optimized_production_full.yml"
echo "GPUs requested: 4 (single node)"
echo "============================================="

# Load modules
module load conda
module load cuda/11.7

# Activate conda environment
source /global/homes/d/dvpatel/.bashrc
conda activate mlmicrophysics-env

# Suppress warnings to reduce log noise
export PYTHONWARNINGS="ignore"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=0

# Change to project directory
cd /global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/pytorch_emulator

# Print system info
echo ""
echo "=== SYSTEM INFO ==="
echo "GPUs allocated:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "==================="
echo ""

# Run the full production job using multi_gpu for single node (matches working debug)
echo "🚀 LAUNCHING FULL PRODUCTION TRAINING"
echo "Config: configs/optimized_production_full.yml"
echo "Mode: Multi-GPU (single node) - same as working debug"
echo "============================================="

# Use multi_gpu flag for single-node multi-GPU training with warning suppression
python -W ignore scripts/train_streaming_parallel.py \
    --config configs/optimized_production_full.yml --multi_gpu

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ FULL PRODUCTION TRAINING COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    echo "   - Model checkpoints: /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/"
    echo "   - Scaler cache: /pscratch/sd/d/dvpatel/mlmicrophysics_project/scaler_cache_optimized/"
    echo ""
    echo "🎯 FULL MICROPHYSICS EMULATOR TRAINING COMPLETED!"
    echo "Ready for evaluation and deployment!"
else
    echo ""
    echo "❌ FULL PRODUCTION TRAINING FAILED"
    echo "Check error logs above for debugging"
    exit 1
fi

# Print completion info
echo ""
echo "============================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds ($((SECONDS/3600)) hours)"
echo "=============================================" 