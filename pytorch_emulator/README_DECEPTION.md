# PyTorch Microphysics Emulator on PNNL Deception — Quickstart

This guide gets you running fast on Deception. For full details on architecture and data loaders, see `pytorch_emulator/README_STREAMING_PARALLEL.md` and `pytorch_emulator/README_DATA_LOADERS.md`.

## Prerequisites
- Deception account and SLURM allocation
- Access to a fast filesystem path for data and outputs
- Module system available (`module`)
- Reference docs: Deception User Guide

## 1) Get the code
```bash
# if not already cloned
git clone https://github.com/dhruvpatel108/mlmicrophysics.git
cd mlmicrophysics
# use the active development branch
git checkout pytorch-emulator
```

## 2) Create and activate the environment (venv; no conda)
Deception discourages conda. Use a Python virtual environment and CUDA 11.8 PyTorch wheels.
```bash
# One-time setup
bash pytorch_emulator/scripts/setup_deception_env.sh

# Activate for interactive/dev work
source ~/.venvs/mlmicrophysics-env/bin/activate
```

## 3) Set paths (adjust for your project)
```bash
export DATA_DIR=/path/to/processed_data
export OUTPUT_DIR=/path/to/outputs
export CHECKPOINT_DIR=/path/to/checkpoints
```

## 4) Update a config for Deception
Pick a base config and update paths. Example keys:
```yaml
# in pytorch_emulator/configs/<your-config>.yml
data:
  data_path: "${DATA_DIR}"
logging:
  output_dir: "${OUTPUT_DIR}"
  checkpoint_dir: "${CHECKPOINT_DIR}"
```
Recommended starting configs:
- `pytorch_emulator/configs/optimized_production_full.yml`
- A multi-GPU config if needed (see configs directory)

## 5) Submit a small test job
Create `run_deception_test.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=mp_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --account=<your-account>
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load cuda/11.8

source ~/.venvs/mlmicrophysics-env/bin/activate

export DATA_DIR=/path/to/processed_data
export OUTPUT_DIR=/path/to/outputs
export CHECKPOINT_DIR=/path/to/checkpoints

cd $SLURM_SUBMIT_DIR
python pytorch_emulator/scripts/train_streaming_parallel.py \
  pytorch_emulator/configs/optimized_streaming_test.yml
```
Submit and monitor:
```bash
sbatch run_deception_test.sbatch
tail -f slurm_<JOBID>.out
```

## Notes specific to Deception
- Prefer venv over conda for environments.
- Use performant storage for `DATA_DIR` and outputs; see the Deception filesystem guidance.
- If using filesystem compression or special paths, consult the system documentation.

## Troubleshooting
- CUDA or driver mismatch: ensure the CUDA module matches your PyTorch build (this guide uses CUDA 11.8 wheels)
- Permission errors: verify `DATA_DIR`, `OUTPUT_DIR`, `CHECKPOINT_DIR` exist and are writable
- OOM on GPU: reduce `batch_size` in your config
- Slow first batch: confirm data path is on performant storage

## Pointers
- Streaming/multi-GPU details: `pytorch_emulator/README_STREAMING_PARALLEL.md`
- Data loader behavior and options: `pytorch_emulator/README_DATA_LOADERS.md`
- Module layout: `pytorch_emulator/readme_module_dependencies.md` 