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
module load cuda/11.8 || true
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
echo "Python version: $(conda run -n a100-ml-env python -V)"
echo "PyTorch and CUDA check:"
# UPDATED: We use conda run to explicitly get info from the correct environment.
conda run -n ml_stable_env python -c "import torch; import sys; print('  Torch:', torch.__version__, '\n  CUDA Available:', torch.cuda.is_available(), '\n  Device Count:', torch.cuda.device_count())"
echo "==================="

# UPDATED: The definitive srun command
# It uses 'conda run' to ensure the correct environment is loaded for the task.
srun conda run -n ml_stable_env python -u scripts/train_streaming_parallel.py \
  --config configs/deception_quick_test.yml

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