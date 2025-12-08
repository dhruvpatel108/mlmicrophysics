#!/bin/bash
#SBATCH --job-name=deeper_heads_files100_subsample1e-1_alpha1e-1_InLog10Standard_OutQuantileRobust
#SBATCH --partition=h100_shared
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=07:00:00
#SBATCH --output=deception_quick_ddp_test-%j.out
#SBATCH --error=deception_quick_ddp_test-%j.err

set -euo pipefail

# --- Configuration Section ---
CONFIG_FILE="configs/deception_quick_test.yml"
# CORRECTED: Use 'conda run' to execute yq from within the environment
OUTPUT_DIR=$(conda run -n pytorch-centos7 yq -r '.data.out_path' "$CONFIG_FILE")
SCALER_DIR=$(conda run -n pytorch-centos7 yq -r '.data.scaler_cache_dir' "$CONFIG_FILE")

# Print job info
echo "🚀 Job Started: $(date)"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================="

# --- Environment Setup ---
# Purge modules to ensure a clean environment
module purge || true
# Source Mamba/Conda from your personal installation
source ~/miniforge3/etc/profile.d/conda.sh

# Set the wandb Run ID to the Slurm Job ID for easy tracking
export WANDB_RUN_ID="deception_$SLURM_JOB_ID"

# Set thread counts for optimal performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Change to the submission directory
cd "$SLURM_SUBMIT_DIR"

# Prepare output directories
echo "## Preparing output directories..."
mkdir -p "${OUTPUT_DIR}/run_${SLURM_JOB_ID}"
mkdir -p "${SCALER_DIR}/run_${SLURM_JOB_ID}"
cp "$CONFIG_FILE" "${OUTPUT_DIR}/run_${SLURM_JOB_ID}/config_used.yml"
echo "## Config file copied to ${OUTPUT_DIR}/run_${SLURM_JOB_ID}/config_used.yml"

# --- System Info & Verification ---
echo " 💻 SYSTEM INFO "
echo "==================="
echo "GPU Info:"
srun nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "PyTorch and CUDA check:"
# Run verification within the correct environment
srun conda run -n pytorch-centos7 python -c "import os, torch; print(f'  Torch: {torch.__version__}\n  CUDA Available: {torch.cuda.is_available()}\n  Device Count: {torch.cuda.device_count()}')"
echo "==================="

# --- Execution ---
echo "## Starting training script..."
# Guard: Fail fast if CUDA is not available
srun conda run -n pytorch-centos7 python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
  echo "❌ CUDA is not available in the pytorch-centos7 environment. Aborting."
  exit 1
fi

# Run the main training command
srun conda run -n pytorch-centos7 python -u scripts/train_streaming_parallel.py \
  --config configs/deception_quick_test.yml --resume /people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_9785956/latest_checkpoint.pth 
  

# --- Completion ---
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed. Check error logs."
    exit 1
fi

echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="