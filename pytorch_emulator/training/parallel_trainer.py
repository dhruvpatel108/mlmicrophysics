"""
Data Parallel Training Pipeline for Constraint-Aware Microphysics Emulator

Implements multi-GPU training to scale up like the original paper (4 V100 GPUs).
Supports both DataParallel (single-node) and DistributedDataParallel (multi-node) training.

Key Features:
- Multi-GPU training with automatic device detection
- DistributedDataParallel for maximum performance
- Gradient accumulation for effective large batch sizes
- Mixed precision training for memory efficiency
- SLURM integration for HPC environments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import GradScaler, autocast

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataParallelTrainer:
    """
    Advanced trainer with multi-GPU support for constraint-aware microphysics emulator.
    
    Features:
    - Automatic multi-GPU detection and setup
    - DistributedDataParallel for optimal performance
    - Mixed precision training with GradScaler
    - Gradient accumulation for large effective batch sizes
    - SLURM and HPC environment integration
    - Robust checkpointing and recovery
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict,
        rank: int = 0,
        world_size: int = 1,
        device: str = "auto"
    ):
        """
        Initialize parallel trainer.
        
        Args:
            model: ConstraintAwareEmulator model
            loss_fn: ConstraintAwareLoss function
            config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
            device: Device specification ('auto', 'cpu', 'cuda', or specific GPU)
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0
        
        # Setup device and distributed training
        self.device = self._setup_device(device)
        self._setup_distributed()
        
        # Setup model with parallel wrapper
        self.model = self._setup_model(model)
        self.loss_fn = loss_fn.to(self.device)
        
        # Training configuration
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 1e-5)
        self.gradient_clip_norm = train_config.get('gradient_clip_norm', 1.0)
        
        # Gradient accumulation for large effective batch sizes
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        
        # Mixed precision training
        self.use_amp = train_config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer(train_config)
        self.scheduler = self._setup_scheduler(train_config)
        
        # Early stopping
        self.early_stopping_patience = train_config.get('early_stopping_patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging (only on main process)
        if self.is_main_process:
            self.setup_logging(config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
        if self.is_main_process:
            logger.info(f"Initialized DataParallelTrainer:")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Distributed: {self.is_distributed}")
            logger.info(f"  Mixed precision: {self.use_amp}")
            logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device."""
        if device == "auto":
            if torch.cuda.is_available():
                if self.is_distributed and 'LOCAL_RANK' in os.environ:
                    device_id = int(os.environ['LOCAL_RANK'])
                    device = torch.device(f'cuda:{device_id}')
                else:
                    device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)
        
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        
        return device
    
    def _setup_distributed(self):
        """Setup distributed training if applicable."""
        if self.is_distributed:
            # Initialize process group
            if not dist.is_initialized():
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(
                    backend=backend,
                    rank=self.rank,
                    world_size=self.world_size
                )
            
            # Synchronize across processes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42 + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + self.rank)
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with appropriate parallel wrapper."""
        model = model.to(self.device)
        
        if self.is_distributed:
            # Use DistributedDataParallel for best performance
            model = DDP(
                model,
                device_ids=[self.device] if self.device.type == 'cuda' else None,
                output_device=self.device if self.device.type == 'cuda' else None,
                find_unused_parameters=False  # Set to True if model has unused params
            )
            if self.is_main_process:
                logger.info("Using DistributedDataParallel")
        elif torch.cuda.device_count() > 1:
            # Use DataParallel for single-node multi-GPU
            model = DP(model)
            if self.is_main_process:
                logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        
        return model
    
    def _setup_optimizer(self, train_config: Dict) -> optim.Optimizer:
        """Setup optimizer with learning rate scaling for distributed training."""
        # Scale learning rate by world size for distributed training
        lr = self.learning_rate
        if self.is_distributed:
            lr *= self.world_size
            if self.is_main_process:
                logger.info(f"Scaled learning rate to {lr} for distributed training")
        
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self, train_config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = train_config.get('scheduler', 'reduce_lr_on_plateau')
        
        if scheduler_name == 'reduce_lr_on_plateau':
            scheduler_params = train_config.get('scheduler_params', {})
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_params.get('mode', 'min'),
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-7)
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 30),
                gamma=train_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def setup_logging(self, config: Dict):
        """Setup logging including W&B (only on main process)."""
        logging_config = config.get('logging', {})
        wandb_config = logging_config.get('wandb', {})
        
        # Initialize W&B only on main process
        if WANDB_AVAILABLE and wandb_config.get('project'):
            try:
                # Add distributed training info to config
                run_config = config.copy()
                run_config['distributed'] = {
                    'world_size': self.world_size,
                    'rank': self.rank,
                    'device': str(self.device)
                }
                
                wandb.init(
                    project=wandb_config.get('project'),
                    entity=wandb_config.get('entity'),
                    group=wandb_config.get('group'),
                    job_type=wandb_config.get('job_type', 'training'),
                    tags=wandb_config.get('tags', []) + [f'rank_{self.rank}'],
                    config=run_config
                )
                self.use_wandb = True
                logger.info("W&B logging initialized")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
        
        # Setup output directory
        self.output_dir = Path(config.get('data', {}).get('out_path', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation and mixed precision."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'classification_loss': [],
            'regression_loss': [],
            'conservation_loss': [],
            'active_samples': []
        }
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(inputs)
                    loss, loss_components = self.loss_fn(predictions, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(inputs)
                loss, loss_components = self.loss_fn(predictions, targets)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping with scaling
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.gradient_clip_norm
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Collect metrics
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            for key, value in loss_components.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value.item())
            
            # Count active samples
            active_count = targets['is_active'].sum().item()
            epoch_metrics['active_samples'].append(active_count)
            
            # Log progress
            if self.is_main_process and batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}, LR: {current_lr:.2e}"
                )
        
        # Synchronize metrics across processes for distributed training
        if self.is_distributed:
            # Average losses across all processes
            epoch_loss_tensor = torch.tensor(epoch_losses).to(self.device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_losses = (epoch_loss_tensor / self.world_size).cpu().tolist()
        
        # Calculate epoch metrics
        metrics = {
            'train_loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'classification_loss': np.mean(epoch_metrics['classification_loss']) if epoch_metrics['classification_loss'] else 0.0,
            'regression_loss': np.mean(epoch_metrics['regression_loss']) if epoch_metrics['regression_loss'] else 0.0,
            'conservation_loss': np.mean(epoch_metrics['conservation_loss']) if epoch_metrics['conservation_loss'] else 0.0,
            'active_fraction': np.mean(epoch_metrics['active_samples']) / inputs.size(0) if epoch_metrics['active_samples'] else 0.0
        }
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        val_metrics = {
            'classification_loss': [],
            'regression_loss': [],
            'conservation_loss': [],
            'active_samples': []
        }
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        predictions = self.model(inputs)
                        loss, loss_components = self.loss_fn(predictions, targets)
                else:
                    predictions = self.model(inputs)
                    loss, loss_components = self.loss_fn(predictions, targets)
                
                # Collect metrics
                val_losses.append(loss.item())
                for key, value in loss_components.items():
                    if key in val_metrics:
                        val_metrics[key].append(value.item())
                
                active_count = targets['is_active'].sum().item()
                val_metrics['active_samples'].append(active_count)
        
        # Synchronize metrics across processes for distributed training
        if self.is_distributed:
            val_loss_tensor = torch.tensor(val_losses).to(self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_losses = (val_loss_tensor / self.world_size).cpu().tolist()
        
        # Calculate validation metrics
        metrics = {
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
            'val_classification_loss': np.mean(val_metrics['classification_loss']) if val_metrics['classification_loss'] else 0.0,
            'val_regression_loss': np.mean(val_metrics['regression_loss']) if val_metrics['regression_loss'] else 0.0,
            'val_conservation_loss': np.mean(val_metrics['conservation_loss']) if val_metrics['conservation_loss'] else 0.0,
            'val_active_fraction': np.mean(val_metrics['active_samples']) / inputs.size(0) if val_metrics['active_samples'] else 0.0
        }
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return
        
        # Get model state dict (unwrap from DDP/DP if needed)
        if isinstance(self.model, (DDP, DP)):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_loss: {metrics['val_loss']:.6f}")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, list]:
        """Complete training loop with multi-GPU support."""
        
        if self.is_main_process:
            logger.info(f"Starting training for {self.epochs} epochs...")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler (if used)
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping check
            val_loss = val_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                is_best = False
            
            # Save checkpoint (only on main process)
            self.save_checkpoint(all_metrics, is_best)
            
            # Logging (only on main process)
            if self.is_main_process:
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.6f}, "
                    f"Val Loss: {val_metrics['val_loss']:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # W&B logging
                if self.use_wandb:
                    wandb.log(all_metrics, step=epoch)
                
                # Store history
                self.train_losses.append(train_metrics['train_loss'])
                self.val_losses.append(val_metrics['val_loss'])
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if self.is_main_process:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        if self.is_main_process:
            logger.info(f"Training completed in {total_time:.2f}s")
            
            # Finalize W&B
            if self.use_wandb:
                wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1
        }
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_training():
    """
    Setup distributed training for SLURM environments.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    # Try SLURM environment variables first
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set master address and port for SLURM
        if 'SLURM_NODELIST' in os.environ:
            import subprocess
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', os.environ['SLURM_NODELIST']],
                stdout=subprocess.PIPE, text=True
            )
            hostnames = result.stdout.strip().split('\n')
            os.environ['MASTER_ADDR'] = hostnames[0]
            os.environ['MASTER_PORT'] = '12355'
    
    # Fall back to torch distributed launch
    elif 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    
    # Single process
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Set local rank environment variable
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    return rank, world_size, local_rank


def create_parallel_trainer(
    model: nn.Module,
    loss_fn: nn.Module,
    config: Dict,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> DataParallelTrainer:
    """
    Create a parallel trainer with automatic distributed setup.
    
    Args:
        model: Model to train
        loss_fn: Loss function
        config: Training configuration
        rank: Process rank (auto-detected if None)
        world_size: Total processes (auto-detected if None)
    
    Returns:
        Configured DataParallelTrainer
    """
    if rank is None or world_size is None:
        rank, world_size, local_rank = setup_distributed_training()
    
    return DataParallelTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        rank=rank,
        world_size=world_size
    )


# Import numpy for metrics calculation
import numpy as np


if __name__ == "__main__":
    """Test distributed training setup."""
    print("🧪 Testing Parallel Training Setup...")
    
    # Test distributed setup
    rank, world_size, local_rank = setup_distributed_training()
    print(f"✅ Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Test device setup
    trainer = DataParallelTrainer(
        model=nn.Linear(10, 1),  # Dummy model
        loss_fn=nn.MSELoss(),    # Dummy loss
        config={'training': {}, 'data': {}},
        rank=rank,
        world_size=world_size
    )
    
    print(f"✅ Trainer device: {trainer.device}")
    print(f"✅ Distributed: {trainer.is_distributed}")
    print(f"✅ Mixed precision: {trainer.use_amp}")
    
    trainer.cleanup()
    print("✅ Parallel training setup test completed!") 