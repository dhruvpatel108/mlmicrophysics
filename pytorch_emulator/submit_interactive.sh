# Print job info
echo "🚀 OPTIMIZED STREAMING DEBUG JOB"
echo "=========================================="
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

# Run the optimized debug job
echo "🚀 LAUNCHING OPTIMIZED STREAMING DEBUG"
echo "=========================================="

#python scripts/train_streaming_parallel.py \
#    --config configs/optimized_debug.yml --multi_gpu
python scripts/train_streaming_parallel.py \
    --config configs/optimized_production_full.yml --multi_gpu


# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ OPTIMIZED DEBUG COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    echo "   - Model checkpoints: /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/"
    echo "   - Scaler cache: /pscratch/sd/d/dvpatel/mlmicrophysics_project/scaler_cache_optimized/"
    echo ""
    echo "🎯 DEBUG VALIDATION PASSED - READY FOR PRODUCTION!"
else
    echo ""
    echo "❌ OPTIMIZED DEBUG FAILED"
    echo "Check error logs above for debugging"
    exit 1
fi

# Print completion info
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "==========================================" 