#!/bin/bash
#SBATCH --job-name=mp_quick_test_distributed
#SBATCH --partition=a100_shared
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # <-- Total number of processes (tasks) to launch
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8              # <-- Request 8 CPUs for each process for data loading
#SBATCH --time=00:20:00
#SBATCH --output=deception_quick_ddp_test-%j.out
#SBATCH --error=deception_quick_ddp_test-%j.err

set -euo pipefail

# --- Configuration Section ---
# Define the config file as the single source of truth
#CONFIG_FILE="configs/deception_quick_test.yml"
# This uses command substitution `$(...)` to run the yq command and assign its output to the OUTPUT_DIR variable.
#OUTPUT_DIR=$(yq -r '.data.out_path' "$CONFIG_FILE")
#SCALER_DIR=$(yq -r '.data.scaler_cache_dir' "$CONFIG_FILE")



# Print job info
echo "🚀 OPTIMIZED STREAMING FULL PRODUCTION JOB"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Current working directory: $(pwd)"
echo "============================================="
module purge || true
module load cuda/11.8 || true

# Use all allocated CPU threads for intra-op parallelism
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}


# --- Environment Setup ---
module load python/miniconda24.1.2
# Initialize the conda command for the script's shell session
source /share/apps/python/miniconda24.1.2/etc/profile.d/conda.sh
# Temporarily disable 'nounset' (`set -u`) for the conda activation,
# as its internal scripts are not fully "strict mode" safe.
set +u
conda activate a100-ml-env
set -u
export MKL_INTERFACE_LAYER=LP64


# Suppress warnings to reduce log noise and improve performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=0

python -V
python -c "import torch; import sys; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
cd "$SLURM_SUBMIT_DIR"

# Prepare output directories
#echo "## Preparing output directories..."
#mkdir -p "$OUTPUT_DIR"
#mkdir -p "$SCALER_DIR"
#cp "$CONFIG_FILE" "${OUTPUT_DIR}/config_used.yml"
#echo "## Config file copied to ${OUTPUT_DIR}/config_used.yml"

# Print system info
echo " 💻 SYSTEM INFO "
echo "==================="
echo "GPUs allocated:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "==================="


#mkdir -p /people/pate014/nersc_mlmicro/outputs/deception_quick_test
#mkdir -p /people/pate014/nersc_mlmicro/scaler_cache/deception_quick_test


#srun python scripts/train_streaming_parallel.py \
#  --config "$CONFIG_FILE" --distributed
srun python scripts/train_streaming_parallel.py \
  --config configs/deception_quick_test.yml 

#status=$?
#echo "Job finished with status $status at $(date)"
#exit $status


# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ FULL PRODUCTION TRAINING COMPLETED SUCCESSFULLY!"
    echo "📊 Check outputs:"
    #echo "   - Model checkpoints: ${OUTPUT_DIR}"
    #echo "   - Scaler cache: ${SCALER_DIR}"
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



# Prefer user's venv named mlmicro; fallback to project venv if needed
#if [ -d "$HOME/.virtualenvs/mlmicro" ]; then
#    source "$HOME/.virtualenvs/mlmicro/bin/activate"
#elif [ -d "$HOME/.venvs/mlmicro" ]; then
#    source "$HOME/.venvs/mlmicro/bin/activate"
#elif [ -d "$HOME/.venvs/mlmicrophysics-env" ]; then
#    source "$HOME/.venvs/mlmicrophysics-env/bin/activate"
#else
#    echo "Virtualenv 'mlmicro' not found in ~/.virtualenvs or ~/.venvs. Aborting."
#    exit 1
#fi