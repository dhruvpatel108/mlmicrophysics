#!/usr/bin/env python3
"""
Streaming + Data Parallel Training Script for Constraint-Aware Microphysics Emulator

Integrates streaming data loading and multi-GPU training to handle large datasets
(1.2B+ samples) with multiple GPUs like the original paper (4 V100 GPUs).

Usage:
    # Single GPU
    python train_streaming_parallel.py --config configs/streaming_parallel_base.yml
    
    # Multi-GPU (DataParallel)
    python train_streaming_parallel.py --config configs/streaming_parallel_base.yml --multi_gpu
    
    # Distributed training (SLURM)
    srun python train_streaming_parallel.py --config configs/streaming_parallel_base.yml --distributed

Features:
- Streaming data loading for memory efficiency
- Multi-GPU training with automatic setup
- Mixed precision training
- SLURM integration for HPC environments
- Comprehensive logging and checkpointing
"""

import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
import sys
import os
import logging
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from models.streaming_data_loader import create_streaming_data_loaders
from training.parallel_trainer import DataParallelTrainer, setup_distributed_training
from models.physics_emulator import ConstraintAwareEmulator
from models.losses import ConstraintAwareLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_loss(config: Dict, device: torch.device) -> tuple:
    """Setup model and loss function."""
    model_config = config['model']
    
    # Create model
    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dim=model_config['head_dim'],
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Create loss function
    loss_fn = ConstraintAwareLoss(
        alpha=model_config.get('alpha', 0.3),
        huber_delta=model_config.get('huber_delta', 1.0),
        conservation_weight=model_config.get('conservation_weight', 0.1),
        use_masking=model_config.get('use_masking', True)
    )
    
    return model, loss_fn


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Streaming + Parallel Training for Microphysics Emulator")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed training (for SLURM/multi-node)'
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        help='Use multi-GPU training on single node'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False
    
    # Setup distributed training if requested
    if args.distributed:
        rank, world_size, local_rank = setup_distributed_training()
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        rank, world_size, local_rank = 0, 1, 0
    
    # Set device
    if torch.cuda.is_available():
        if args.distributed:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Only log from main process in distributed training
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info("🚀 STREAMING + PARALLEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"📁 Config: {args.config}")
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"🔄 Distributed: {args.distributed}")
        logger.info(f"🎯 Multi-GPU: {args.multi_gpu}")
        logger.info(f"🌍 World size: {world_size}")
        logger.info("=" * 60)
    
    try:
        # Setup data loaders with streaming
        if is_main_process:
            logger.info("📊 Setting up streaming data loaders...")
        
        data_config = config['data']
        train_loader, val_loader, scaler = create_streaming_data_loaders(
            data_path=data_config['data_path'],
            config=config,
            train_fraction=data_config.get('train_fraction', 0.8),
            batch_size=data_config.get('batch_size', 1024),
            num_workers=data_config.get('num_workers', 0),
            scaler_cache_dir=data_config.get('scaler_cache_dir', './scaler_cache')
        )
        
        if is_main_process:
            logger.info("✅ Streaming data loaders created successfully")
            # Print dataset lengths
            logger.info(f"📊 Dataset sizes:")
            logger.info(f"   Training dataset: {len(train_loader.dataset):,} samples")
            logger.info(f"   Validation dataset: {len(val_loader.dataset):,} samples")
            logger.info(f"   Batch size: {data_config.get('batch_size', 1024)}")
            logger.info(f"   Training batches: {len(train_loader)}")
            logger.info(f"   Validation batches: {len(val_loader)}")
        # Setup model and loss
        if is_main_process:
            logger.info("🧠 Setting up model and loss function...")
        
        model, loss_fn = setup_model_and_loss(config, device)
        
        if is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Setup trainer with parallel support
        if is_main_process:
            logger.info("🏃‍♂️ Setting up parallel trainer...")
        
        trainer = DataParallelTrainer(
            model=model,
            loss_fn=loss_fn,
            config=config,
            rank=rank,
            world_size=world_size,
            device="auto"
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            if is_main_process:
                logger.info(f"📂 Resuming from checkpoint: {args.resume}")
            
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state (handle DDP/DP wrapper)
            if hasattr(trainer.model, 'module'):
                trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler state
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if trainer.scaler and checkpoint.get('scaler_state_dict'):
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Load training state
            trainer.epoch = checkpoint.get('epoch', 0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            if is_main_process:
                logger.info(f"✅ Resumed from epoch {trainer.epoch + 1}")
        
        # Start training
        if is_main_process:
            logger.info("🎯 Starting training...")
            logger.info("=" * 60)
        
        # Run training
        history = trainer.train(train_loader, val_loader)
        
        if is_main_process:
            logger.info("=" * 60)
            logger.info("🎉 TRAINING COMPLETED!")
            logger.info(f"   Total epochs: {history['total_epochs']}")
            logger.info(f"   Best validation loss: {history['best_val_loss']:.6f}")
            logger.info(f"   Final training loss: {history['train_losses'][-1]:.6f}")
            logger.info(f"   Checkpoints saved to: {trainer.output_dir}")
            logger.info("=" * 60)
        
        # Cleanup distributed training
        trainer.cleanup()
        
        return True
        
    except KeyboardInterrupt:
        if is_main_process:
            logger.info("\n⚠️ Training interrupted by user")
        return False
        
    except Exception as e:
        if is_main_process:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        return False
    
    finally:
        # Final cleanup
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


def validate_config(config: Dict) -> bool:
    """Validate configuration file."""
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required section '{section}' in config")
            return False
    
    # Validate model config
    model_config = config['model']
    required_model_keys = ['input_dim', 'shared_dims', 'head_dim']
    for key in required_model_keys:
        if key not in model_config:
            logger.error(f"Missing required model parameter '{key}'")
            return False
    
    # Validate data config
    data_config = config['data']
    required_data_keys = ['data_path', 'input_cols', 'output_cols']
    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"Missing required data parameter '{key}'")
            return False
    
    # Check if data path exists
    data_path = Path(data_config['data_path'])
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return False
    
    return True


def create_slurm_script(config_path: str, output_path: str = "submit_streaming_parallel.sh"):
    """Create a SLURM submission script for distributed training."""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=streaming_parallel_training
#SBATCH --account=m4942
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=2                     # Use 2 nodes for distributed training
#SBATCH --ntasks-per-node=4          # 4 GPUs per node = 8 total GPUs
#SBATCH --cpus-per-task=8            # 8 CPUs per GPU
#SBATCH --gpus-per-node=4            # 4 GPUs per node (A100s)
#SBATCH --mem=240G                   # 240GB memory per node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules
module load python
module load cuda/11.7

# Activate conda environment
conda activate mlmicrophysics-env

# Change to project directory
cd /global/homes/d/dvpatel/mlmicrophysics_project/mlmicrophysics/pytorch_emulator

# Print system info
echo "=== SYSTEM INFO ==="
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "==================="

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run distributed training
echo "🚀 STREAMING + DISTRIBUTED PARALLEL TRAINING"
echo "Config: {config_path}"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: 8"
echo "Expected: Massive speedup for large datasets"
echo "==========================================="

srun python scripts/train_streaming_parallel.py \\
    --config {config_path} \\
    --distributed

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Check outputs in the configured output directory"
else
    echo "❌ Training failed"
    exit 1
fi

# Print completion info
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    logger.info(f"Created SLURM script: {output_path}")
    logger.info("Submit with: sbatch submit_streaming_parallel.sh")


if __name__ == "__main__":
    # Test if running as main script
    if len(sys.argv) > 1 and sys.argv[1] == "--create-slurm":
        # Create SLURM script mode
        config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/streaming_parallel_base.yml"
        create_slurm_script(config_path)
    else:
        # Normal training mode
        success = main()
        sys.exit(0 if success else 1) 