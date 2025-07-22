#!/bin/bash
#SBATCH --job-name=overfitting_test
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=debug                  # Use debug queue for quick testing
#SBATCH --time=0:30:00               # 30 minute time limit
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks-per-node=1          # Single task
#SBATCH --cpus-per-task=4            # 4 CPUs
#SBATCH --gpus-per-node=1            # Single A100 GPU
#SBATCH --mem=32G                    # 32GB memory
#SBATCH --output=overfitting_test-%j.out
#SBATCH --error=overfitting_test-%j.err

# Print job info
echo "🧪 OVERFITTING TEST JOB (LARGE MODEL)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Expected duration: 5-15 minutes"
echo "GPUs requested: 1"
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
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "==================="
echo ""

# Run the overfitting test
echo "🚀 LAUNCHING OVERFITTING TEST"
echo "Config: configs/overfitting_large_model.yml"
echo "=========================================="

python scripts/train_streaming_parallel.py \
    --config configs/overfitting_large_model.yml

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ OVERFITTING TEST COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    echo "   - Model checkpoints: /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_overfitting_outputs/"
    echo "   - Scaler cache: /pscratch/sd/d/dvpatel/mlmicrophysics_project/scaler_cache_overfitting/"
    echo ""
    echo "🎯 OVERFITTING PIPELINE VALIDATED!"
    echo "   Large model training confirmed working"
else
    echo ""
    echo "❌ OVERFITTING TEST FAILED"
    echo "Check error logs above for debugging"
    exit 1
fi

# Print completion info
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "==========================================" 