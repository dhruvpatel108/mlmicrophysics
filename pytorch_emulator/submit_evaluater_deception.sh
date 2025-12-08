#!/bin/bash
#SBATCH --job-name=mp_quick_test_distributed_condarun
#SBATCH --partition=a100_shared
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8             
#SBATCH --time=00:20:00
#SBATCH --output=evaluater_deception_quick_test-%j.out
#SBATCH --error=evaluater_deception_quick_test-%j.err


set -euo pipefail

# --- Environment Setup ---
module purge || true
module load gcc/11.5.0
module load cuda/12.9.1
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
# Use all allocated CPU threads for intra-op parallelism
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Suppress warnings to reduce log noise and improve performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=0
cd "$SLURM_SUBMIT_DIR"

# --- Configuration Section ---
#CONFIG_FILE="$SLURM_SUBMIT_DIR/configs/filtered_data_runs.yml"
CONFIG_FILE="/people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_107666/config_used_quick_eval.yml"
CHECKPOINT_PATH="/people/pate014/nersc_mlmicro/outputs/deception_distributed_test/run_107666/latest_checkpoint.pth"
EVAL_OUTPUT_DIR="/people/pate014/nersc_mlmicro/mlmicrophysics/pytorch_emulator/evaluation_results/run_107666_log_printing_files762_subsample0.1"
OUTPUT_DIR=$(conda run -n rocky9-pytorch yq -r '.data.out_path' "$CONFIG_FILE")
SCALER_DIR=$(conda run -n rocky9-pytorch yq -r '.data.scaler_cache_dir' "$CONFIG_FILE")

mkdir -p "$EVAL_OUTPUT_DIR"



# Print system info
echo " 💻 SYSTEM INFO "
echo "==================="
echo "GPUs allocated:"
# UPDATED: We use a separate srun/conda run command for nvidia-smi.
srun nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Python version: $(conda run -n rocky9-pytorch python -V)"
echo "PyTorch and CUDA check:"
# UPDATED: We use conda run to explicitly get info from the correct environment.
conda run -n rocky9-pytorch python -c "import torch; import sys; print('  Torch:', torch.__version__, '\n  CUDA Available:', torch.cuda.is_available(), '\n  Device Count:', torch.cuda.device_count())"
echo "==================="


srun conda run -n rocky9-pytorch python scripts/evaluate_model_debug.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_FILE" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --num_random_samples 2000 \
  --scaler_run_id run_107666 \
  --evaluation_space log

#python scripts/evaluate_model.py \
#--checkpoint /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/run_41230175/best_checkpoint.pth \
#--config configs/optimized_production_full.yml \
#--output_dir evaluation_results/run41230175






