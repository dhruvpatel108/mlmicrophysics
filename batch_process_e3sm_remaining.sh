#!/bin/bash
#SBATCH --job-name=process_e3sm_12.3
#SBATCH --account=m4942
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --licenses=cfs,SCRATCH
#SBATCH --output=process_e3sm_12.3_%j.out
#SBATCH --error=process_e3sm_12.3_%j.err

# Load required modules
module load conda

# Activate conda environment
conda activate mlmicrophysics-env

# Change to scripts directory
cd /global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/scripts

# Run E3SM processing for all 7 files of version 12.3 data
echo "Starting E3SM 12.3 data processing for ALL FILES at $(date)"

# Set memory-conservative environment variables
export OMP_NUM_THREADS=4
export DASK_ARRAY__SLICING__SPLIT_LARGE_CHUNKS=True

python process_e3sm_output_remaining.py ../config/e3sm_tau_run1_process_remaining.yml -p 1

echo "E3SM 12.3 data processing completed at $(date)"

# Check processed data
echo "Checking processed data:"
ls -lh /pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_e3sm300_mlmicro12.3_tau_run/ | tail -20

echo "Processing of 12.3 files completed successfully!" 