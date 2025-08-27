#!/bin/bash
#SBATCH --job-name=mp_quick_test_distributed_condarun
#SBATCH --partition=a100_shared
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # <-- Total number of processes (tasks) to launch
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8              # <-- Request 8 CPUs for each process for data loading
#SBATCH --time=00:20:00
#SBATCH --output=evaluater_deception_quick_test-%j.out
#SBATCH --error=evaluater_deception_quick_test-%j.err


set -euo pipefail
# --- Configuration Section ---
# UPDATED: Define the config file as the single source of truth.
CONFIG_FILE="configs/deception_quick_test.yml"
# UPDATED: This uses command substitution and yq to get output paths.
OUTPUT_DIR=$(yq -r '.data.out_path' "$CONFIG_FILE")
SCALER_DIR=$(yq -r '.data.scaler_cache_dir' "$CONFIG_FILE") 

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



# Print system info
echo " 💻 SYSTEM INFO "
echo "==================="
echo "GPUs allocated:"
# UPDATED: We use a separate srun/conda run command for nvidia-smi.
srun nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(conda run -n ml_stable_env python -V)"
echo "PyTorch and CUDA check:"
# UPDATED: We use conda run to explicitly get info from the correct environment.
conda run -n ml_stable_env python -c "import torch; import sys; print('  Torch:', torch.__version__, '\n  CUDA Available:', torch.cuda.is_available(), '\n  Device Count:', torch.cuda.device_count())"
echo "==================="


srun conda run -n ml_stable_env python scripts/evaluate_model.py \
--checkpoint /people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_9650929/best_checkpoint.pth \
--config configs/deception_quick_test.yml \
--output_dir evaluation_results/run_9650929

#python scripts/evaluate_model.py \
#--checkpoint /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/run_41230175/best_checkpoint.pth \
#--config configs/optimized_production_full.yml \
#--output_dir evaluation_results/run41230175






