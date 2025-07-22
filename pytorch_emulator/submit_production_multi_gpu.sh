#!/bin/bash
#SBATCH --job-name=multi_gpu_production
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=regular                    # Use regular queue for production
#SBATCH --time=4:00:00                 # 2 hour time limit
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks-per-node=1            # Single task
#SBATCH --cpus-per-task=8              # 8 CPUs for multi-GPU
#SBATCH --gpus-per-node=4              # 4 A100 GPUs for multi-GPU training
#SBATCH --mem=64G                      # 64GB memory for multi-GPU
#SBATCH --output=multi_gpu_production-%j.out
#SBATCH --error=multi_gpu_production-%j.err

# Print job info
echo "🚀 MULTI-GPU PRODUCTION JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPUs requested: 4"
echo "=========================================="

# Load modules
module load conda
module load cuda/11.7

# Activate conda environment
source /global/homes/d/dvpatel/.bashrc
conda activate mlmicrophysics-env

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

# Run the multi-GPU production job
echo "🚀 LAUNCHING MULTI-GPU PRODUCTION"
echo "Config: configs/multi_gpu_production.yml"
echo "=========================================="

python scripts/train_streaming_parallel.py \
    --config configs/multi_gpu_production.yml --multi_gpu

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MULTI-GPU PRODUCTION COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    echo "   - Model checkpoints: /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/"
    echo "   - Scaler cache: /pscratch/sd/d/dvpatel/mlmicrophysics_project/scaler_cache_test/"
    echo ""
    echo "🎯 MULTI-GPU PRODUCTION PIPELINE COMPLETED!"
else
    echo ""
    echo "❌ MULTI-GPU PRODUCTION FAILED"
    echo "Check error logs above for debugging"
    exit 1
fi

# Print completion info
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "==========================================" 