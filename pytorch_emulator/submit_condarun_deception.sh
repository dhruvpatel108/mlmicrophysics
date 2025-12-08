#!/bin/bash
#SBATCH --job-name=mp_quick_test_distributed_condarun
#SBATCH --partition=a100_shared
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=32              
#SBATCH --time=05:00:00
#SBATCH --output=deception_quick_ddp_test-%j.out
#SBATCH --error=deception_quick_ddp_test-%j.err

set -euo pipefail

# --- Configuration Section ---
# UPDATED: Define the config file as the single source of truth.
CONFIG_FILE="configs/deception_quick_test.yml"
# UPDATED: This uses command substitution and yq to get output paths.
OUTPUT_DIR=$(yq -r '.data.out_path' "$CONFIG_FILE")
SCALER_DIR=$(yq -r '.data.scaler_cache_dir' "$CONFIG_FILE")


# Print job info
echo "🚀 OPTIMIZED STREAMING FULL PRODUCTION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Current working directory: $(pwd)"
echo "============================================="

# UPDATED: Simplified Environment Setup.
# We only load modules and source the conda command;
# the environment activation is handled by `conda run`.
module purge || true
# Avoid loading a specific CUDA module to prevent library conflicts with PyTorch's CUDA runtime
# module load cuda/11.8 || true
module load python/miniconda24.1.2
source /share/apps/python/miniconda24.1.2/etc/profile.d/conda.sh

# Use all allocated CPU threads for intra-op parallelism
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Suppress warnings to reduce log noise and improve performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=0

cd "$SLURM_SUBMIT_DIR"

# Prepare output directories
echo "## Preparing output directories..."
# UPDATED: These lines are now uncommented and use the yq variables.
mkdir -p "${OUTPUT_DIR}/run_${SLURM_JOB_ID}"
mkdir -p "${SCALER_DIR}/run_${SLURM_JOB_ID}"
cp "$CONFIG_FILE" "${OUTPUT_DIR}/run_${SLURM_JOB_ID}/config_used.yml"
echo "## Config file copied to ${OUTPUT_DIR}/run_${SLURM_JOB_ID}/config_used.yml"

# Print system info
echo " 💻 SYSTEM INFO "
echo "==================="
echo "GPUs allocated:"
# UPDATED: We use a separate srun/conda run command for nvidia-smi.
srun nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(conda run -n ml_centos7_robust_env python -V)"
echo "PyTorch and CUDA check:"
# UPDATED: Run under srun to inherit SLURM's GPU binding and print more diagnostics.
srun conda run -n ml_centos7_robust_env python -c "import os, torch; print('  Torch:', torch.__version__, '\n  Torch CUDA:', torch.version.cuda, '\n  CUDA Built:', torch.backends.cuda.is_built(), '\n  CUDA Available:', torch.cuda.is_available(), '\n  Device Count:', torch.cuda.device_count(), '\n  CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))"
echo "==================="

# UPDATED: The definitive srun command
# It uses 'conda run' to ensure the correct environment is loaded for the task.
# Guard: fail fast if CUDA is not available to avoid CPU fallback
srun conda run -n ml_centos7_robust_env python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)"

if [ $? -ne 0 ]; then
  echo "❌ CUDA is not available in the selected environment (ml_stable_env). Aborting training."
  echo "   Hint: Ensure the environment has a CUDA-enabled PyTorch (pytorch-cuda=12.x)."
  exit 2
fi

srun conda run -n ml_centos7_robust_env python -u scripts/train_streaming_parallel.py \
  --config configs/deception_quick_test.yml --resume /people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_9659273/best_checkpoint.pth --device cuda

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ FULL PRODUCTION TRAINING COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    # UPDATED: These lines are now uncommented and use the yq variables.
    echo "   - Model checkpoints: ${OUTPUT_DIR}"
    echo "   - Scaler cache: ${SCALER_DIR}"
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