#!/bin/bash
#SBATCH --account=pioneercloud
#SBATCH --partition=h100_shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#
#

# --- ENVIRONMENT SETUP (MOVED TO TOP) ---
set -euo pipefail
echo "🚀 Job Started: $(date)"
echo "Loading system modules for job $SLURM_JOB_ID..."

# 1. Purge any modules
module purge

# 2. Load the system's Rocky 9 modules
module load gcc/11.5.0
module load cuda/12.9.1
module load python/miniconda25.5.1

# 3. Source the conda script to make 'conda' command available
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
echo "System modules and conda loaded."
# --- END SETUP ---

# --- CONFIGURATION SECTION (NOW SAFE TO RUN) ---
# Set the path to your config file
#CONFIG_FILE="configs/filtered_data_runs.yml"
#CONFIG_FILE="configs/filtered_data_smoketest.yml"
CONFIG_FILE="/people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_107666/config_used.yml"


# Now that conda is loaded, we can safely run these commands
JOB_NAME=$(conda run -n rocky9-pytorch yq -r '.experiment.name' "$CONFIG_FILE")
OUTPUT_DIR=$(conda run -n rocky9-pytorch yq -r '.data.out_path' "$CONFIG_FILE")
SCALER_DIR=$(conda run -n rocky9-pytorch yq -r '.data.scaler_cache_dir' "$CONFIG_FILE")

# --- RE-SET DYNAMIC SBATCH OPTIONS ---
# This is a good practice if you want dynamic job names/outputs
# Note: These lines are informational for sbatch, which has already read them.
# The main benefit is that your job name is set in $JOB_NAME
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${JOB_NAME}-%j.out
#SBATCH --error=${JOB_NAME}-%j.err

# Print job info
echo "============================================="
echo "Requested partition: ${SLURM_JOB_PARTITION}"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Config File: $CONFIG_FILE"
echo "============================================="

# Set thread counts and wandb ID
export WANDB_RUN_ID="deception_$SLURM_JOB_ID"
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
srun --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "PyTorch and CUDA check:"
srun --ntasks=1 conda run -n rocky9-pytorch python -c "import os, torch; print(f'  Torch: {torch.__version__}\n  CUDA Available: {torch.cuda.is_available()}\n  Device Count: {torch.cuda.device_count()}')"
echo "==================="

# --- Execution ---
echo "## Starting training script..."
# Guard: Fail fast if CUDA is not available
srun conda run -n rocky9-pytorch python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
  echo "❌ CUDA is not available in the rocky9-pytorch environment. Aborting."
  exit 1
fi

# Run the main training command
srun conda run -n rocky9-pytorch python -u scripts/train_streaming_parallel.py \
  --config $CONFIG_FILE --distributed --resume /people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_107666/latest_checkpoint.pth --scaler_run_id 107666
  
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