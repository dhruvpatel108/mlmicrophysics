#!/bin/bash
#SBATCH --job-name=mp_quick_test_distributed_condarun
#SBATCH --partition=dlt
#SBATCH --account=pioneercloud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # <-- Total number of processes (tasks) to launch
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8              # <-- Request 8 CPUs for each process for data loading
#SBATCH --time=00:20:00
#SBATCH --output=env_test-%j.out
#SBATCH --error=env_test-%j.err

echo "--- Starting Environment Test ---"

# 1. Load the Miniconda module
module load python/miniconda24.1.2

# 2. Initialize Conda
source /share/apps/python/miniconda24.1.2/etc/profile.d/conda.sh

# 3. Activate your environment
conda activate ml_stable_env

# 4. Run the verification command
echo "Running Python verification..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU found')"

echo "--- Test Finished ---"