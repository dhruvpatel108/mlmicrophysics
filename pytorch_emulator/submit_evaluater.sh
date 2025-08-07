#!/bin/bash
#SBATCH --job-name=optimized_debug
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=debug                    # Use debug queue for quick testing
#SBATCH --time=0:30:00                 # 30 minute time limit
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks-per-node=1            # Single task
#SBATCH --cpus-per-task=16              # 8 CPUs for multi-GPU
#SBATCH --gpus-per-node=1              # 4 A100 GPUs for multi-GPU training
#SBATCH --mem=128G                      # 64GB memory for multi-GPU
#SBATCH --output=evaluate_model-%j.out
#SBATCH --error=evaluate_model-%j.err



python scripts/evaluate_model.py \
--checkpoint /pscratch/sd/d/dvpatel/mlmicrophysics_project/pytorch_multi_gpu_outputs/run_41230175/best_checkpoint.pth \
--config configs/optimized_production_full.yml \
--output_dir evaluation_results/run41230175



