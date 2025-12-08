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
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import GradScaler#, autocast
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
        device: str = "auto",
        use_data_parallel: bool = False
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
            use_data_parallel: Enable `nn.DataParallel` when multiple GPUs are visible
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_data_parallel = use_data_parallel
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
        self.epochs = int(train_config.get('epochs', 100))
        self.learning_rate = float(train_config.get('learning_rate', 0.001))
        self.weight_decay = float(train_config.get('weight_decay', 1e-5))
        self.gradient_clip_norm = float(train_config.get('gradient_clip_norm', 1.0))
        
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
        
        # Logging frequency and W&B image cadence
        logging_config = config.get('logging', {})
        self.log_frequency = int(logging_config.get('log_frequency', 50))
        self.epoch_log_frequency = int(logging_config.get('epoch_log_frequency', 1))
        self.image_log_percent = float(logging_config.get('image_log_percent', 5))
        self.image_log_interval = max(1, int(self.epochs * (self.image_log_percent / 100.0)))
        wandb_config = config.get('wandb', {})
        self.wandb_max_val_batches = int(wandb_config.get('max_val_batches', 8))
        if self.wandb_max_val_batches < 0:
            self.wandb_max_val_batches = 0
        
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
                    device = torch.device('cuda:0')  
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)
        
        if device.type == 'cuda':
            # Ensure we have a specific device index for set_device
            if device.index is None:
                device = torch.device(f'cuda:{device.index or 0}')
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
            device_ids = None
            output_device = None
            if self.device.type == 'cuda':
                if self.device.index is None:
                    raise RuntimeError("Expected CUDA device index to be set when using distributed training")
                device_ids = [self.device.index]
                output_device = self.device.index
            model = DDP(
                model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=False  # Set to True if model has unused params
            )
            if self.is_main_process:
                logger.info("Using DistributedDataParallel")
        elif self.use_data_parallel and torch.cuda.device_count() > 1:
            # Use DataParallel for single-node multi-GPU
            model = DP(model)
            if self.is_main_process:
                logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        elif self.use_data_parallel and torch.cuda.device_count() <= 1 and self.is_main_process:
            logger.warning("DataParallel requested but fewer than 2 CUDA devices detected. Continuing with single device.")
        
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
                factor=float(scheduler_params.get('factor', 0.5)),
                patience=int(scheduler_params.get('patience', 5)),
                min_lr=float(scheduler_params.get('min_lr', 1e-7))
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(train_config.get('step_size', 30)),
                gamma=float(train_config.get('gamma', 0.1))
            )
        else:
            return None

    def _set_epoch_for_loader(self, loader: DataLoader, epoch: int):
        """Safely set epoch for loaders that expose a sampler."""
        dataset = getattr(loader, 'dataset', None)
        if isinstance(dataset, IterableDataset):
            return
        sampler = None
        try:
            sampler = loader.sampler  # May raise for iterable datasets
        except (AttributeError, TypeError, ValueError):
            sampler = None

        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
    
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
        loss_sum = 0.0
        loss_count = 0.0
        metric_sums = {
            'classification_loss': 0.0,
            'regression_loss': 0.0,
            'conservation_loss': 0.0
        }
        metric_counts = {
            'classification_loss': 0.0,
            'regression_loss': 0.0,
            'conservation_loss': 0.0
        }
        active_total = 0.0
        sample_total = 0.0
        
        self.optimizer.zero_grad()
        #logger.info(f"ckpt: pre batch loop")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            t_batch_start = time.time()
            #logger.info(f"Training batch_idx: {batch_idx}")
            # Move to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            # Print train batch class balance for first few batches
            if self.is_main_process and batch_idx < 5 and 'is_active' in targets:
                try:
                    _bsz = targets['is_active'].numel()
                    _act = int(targets['is_active'].sum().item())
                    _inact = _bsz - _act
                    print(f"[train] epoch {self.epoch} batch {batch_idx}: active={_act/_bsz:.2%}, inactive={_inact/_bsz:.2%}")
                except Exception:
                    pass
            
            # Forward pass with mixed precision
            if self.use_amp:
                #with autocast():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = self.model(inputs)
                    loss_dict = self.loss_fn(predictions, targets)
                    loss = loss_dict['total_loss']
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(inputs)
                loss_dict = self.loss_fn(predictions, targets)
                loss = loss_dict['total_loss']
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
            loss_value = loss.item() * self.gradient_accumulation_steps
            loss_sum += loss_value
            loss_count += 1.0
            for key, value in loss_dict.items():
                if key in metric_sums:
                    metric_value = value.item() if hasattr(value, 'item') else float(value)
                    metric_sums[key] += metric_value
                    metric_counts[key] += 1.0
            
            # Count active samples
            if 'is_active' in targets:
                active_count = targets['is_active'].sum().item()
                active_total += active_count
                sample_total += targets['is_active'].numel()
            
            # Log progress
            if self.is_main_process and self.log_frequency > 0 and batch_idx % self.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Detailed loss component logging
                cls_loss = loss_dict.get('classification_loss', 0.0)
                reg_loss = loss_dict.get('regression_loss', 0.0)
                active_count = loss_dict.get('active_samples', 0)
                total_samples = inputs.size(0)
                active_fraction = active_count / total_samples if total_samples > 0 else 0.0
                
                # Convert to float if they're tensors
                if hasattr(cls_loss, 'item'):
                    cls_loss = cls_loss.item()
                if hasattr(reg_loss, 'item'):
                    reg_loss = reg_loss.item()
                
                # Use batch_idx + 1 to show current batch number (1-indexed)
                logger.info(
                    f"Epoch {self.epoch+1}, Batch {batch_idx + 1}, "
                    f"Loss: {loss.item():.6f}, LR: {current_lr:.2e}, "
                    f"CLS: {cls_loss:.4f}, REG: {reg_loss:.4f}, "
                    f"Active: {active_count}/{total_samples} ({active_fraction:.2%})"
                )
            t_batch_end = time.time()
            #logger.info(f"Batch {batch_idx + 1} took {t_batch_end - t_batch_start:.2f} seconds")
        # Synchronize metrics across processes for distributed training
        if self.is_distributed:
            stats = torch.tensor(
                [
                    loss_sum,
                    loss_count,
                    metric_sums['classification_loss'],
                    metric_counts['classification_loss'],
                    metric_sums['regression_loss'],
                    metric_counts['regression_loss'],
                    metric_sums['conservation_loss'],
                    metric_counts['conservation_loss'],
                    active_total,
                    sample_total
                ],
                dtype=torch.float64,
                device=self.device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            (
                loss_sum,
                loss_count,
                metric_sums['classification_loss'],
                metric_counts['classification_loss'],
                metric_sums['regression_loss'],
                metric_counts['regression_loss'],
                metric_sums['conservation_loss'],
                metric_counts['conservation_loss'],
                active_total,
                sample_total
            ) = stats.tolist()
        
        # Calculate epoch metrics
        metrics = {
            'train_loss': (loss_sum / loss_count) if loss_count > 0 else 0.0,
            'classification_loss': (metric_sums['classification_loss'] / metric_counts['classification_loss']) if metric_counts['classification_loss'] > 0 else 0.0,
            'regression_loss': (metric_sums['regression_loss'] / metric_counts['regression_loss']) if metric_counts['regression_loss'] > 0 else 0.0,
            'conservation_loss': (metric_sums['conservation_loss'] / metric_counts['conservation_loss']) if metric_counts['conservation_loss'] > 0 else 0.0,
            'active_fraction': (active_total / sample_total) if sample_total > 0 else 0.0
        }
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loss_sum = 0.0
        val_loss_count = 0.0
        val_metric_sums = {
            'classification_loss': 0.0,
            'regression_loss': 0.0,
            'conservation_loss': 0.0
        }
        val_metric_counts = {
            'classification_loss': 0.0,
            'regression_loss': 0.0,
            'conservation_loss': 0.0
        }
        val_active_total = 0.0
        val_sample_total = 0.0
        
        with torch.no_grad():
            for bidx, (inputs, targets) in enumerate(val_loader):
                # Move to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                # Print val batch class balance for first few batches
                if self.is_main_process and bidx < 5 and 'is_active' in targets:
                    try:
                        _bsz = targets['is_active'].numel()
                        _act = int(targets['is_active'].sum().item())
                        _inact = _bsz - _act
                        print(f"[val]   epoch {self.epoch} batch {bidx}: active={_act/_bsz:.2%}, inactive={_inact/_bsz:.2%}")
                    except Exception:
                        pass
                
                # Forward pass
                if self.use_amp:
                    #with autocast():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.model(inputs)
                        loss_dict = self.loss_fn(predictions, targets)
                else:
                    predictions = self.model(inputs)
                    loss_dict = self.loss_fn(predictions, targets)
                
                # Collect metrics
                loss_val = loss_dict['total_loss'].item()
                val_loss_sum += loss_val
                val_loss_count += 1.0
                for key, value in loss_dict.items():
                    if key in val_metric_sums:
                        metric_value = value.item() if hasattr(value, 'item') else float(value)
                        val_metric_sums[key] += metric_value
                        val_metric_counts[key] += 1.0
                
                if 'is_active' in targets:
                    active_count = targets['is_active'].sum().item()
                    val_active_total += active_count
                    val_sample_total += targets['is_active'].numel()
        
        # Synchronize metrics across processes for distributed training
        if self.is_distributed:
            val_stats = torch.tensor(
                [
                    val_loss_sum,
                    val_loss_count,
                    val_metric_sums['classification_loss'],
                    val_metric_counts['classification_loss'],
                    val_metric_sums['regression_loss'],
                    val_metric_counts['regression_loss'],
                    val_metric_sums['conservation_loss'],
                    val_metric_counts['conservation_loss'],
                    val_active_total,
                    val_sample_total
                ],
                dtype=torch.float64,
                device=self.device
            )
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            (
                val_loss_sum,
                val_loss_count,
                val_metric_sums['classification_loss'],
                val_metric_counts['classification_loss'],
                val_metric_sums['regression_loss'],
                val_metric_counts['regression_loss'],
                val_metric_sums['conservation_loss'],
                val_metric_counts['conservation_loss'],
                val_active_total,
                val_sample_total
            ) = val_stats.tolist()
        
        # Calculate validation metrics
        metrics = {
            'val_loss': (val_loss_sum / val_loss_count) if val_loss_count > 0 else 0.0,
            'val_classification_loss': (val_metric_sums['classification_loss'] / val_metric_counts['classification_loss']) if val_metric_counts['classification_loss'] > 0 else 0.0,
            'val_regression_loss': (val_metric_sums['regression_loss'] / val_metric_counts['regression_loss']) if val_metric_counts['regression_loss'] > 0 else 0.0,
            'val_conservation_loss': (val_metric_sums['conservation_loss'] / val_metric_counts['conservation_loss']) if val_metric_counts['conservation_loss'] > 0 else 0.0,
            'val_active_fraction': (val_active_total / val_sample_total) if val_sample_total > 0 else 0.0
        }
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return
        import os
        from datetime import datetime
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
        # Unique run directory: use SLURM_JOB_ID if available, else timestamp
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id is None:
            job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.output_dir / f"run_{job_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save latest checkpoint
        checkpoint_path = run_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        # Save best checkpoint
        if is_best:
            best_path = run_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_loss: {metrics['val_loss']:.6f} in {run_dir}")
    
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
            logger.info(f"Epoch {epoch}...")
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler (if used)
            self._set_epoch_for_loader(train_loader, epoch)
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

            if self.is_distributed and dist.is_initialized():
                best_tensor = torch.tensor(self.best_val_loss, dtype=torch.float64, device=self.device)
                dist.broadcast(best_tensor, src=0)
                self.best_val_loss = best_tensor.item()
            
            # Save checkpoint (only on main process)
            self.save_checkpoint(all_metrics, is_best)

            # W&B: epoch-level scalar metrics and full-validation scatter image (only main process)
            if self.is_main_process and self.use_wandb and (epoch % self.epoch_log_frequency == 0):
                # Compute full-validation predictions and metrics
                try:
                    import numpy as np
                    from sklearn.metrics import r2_score
                    import matplotlib.pyplot as plt
                    self.model.eval()
                    preds_all = {k: [] for k in ['qrtend', 'nctend', 'nrtend', 'qctend', 'is_active']}
                    trues_all = {k: [] for k in ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU', 'is_active']}
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(val_loader):
                            inputs = inputs.to(self.device, non_blocking=True)
                            targ_dev = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                            out = self.model(inputs)
                            # Collect
                            for key in ['qrtend', 'nctend', 'nrtend', 'qctend', 'is_active']:
                                if key in out:
                                    preds_all[key].append(out[key].detach().float().view(-1).cpu().numpy())
                            for key in ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU', 'is_active']:
                                if key in targ_dev:
                                    trues_all[key].append(targ_dev[key].detach().float().view(-1).cpu().numpy())
                            if self.wandb_max_val_batches and (batch_idx + 1) >= self.wandb_max_val_batches:
                                break
                    # Concatenate
                    for k in preds_all:
                        if preds_all[k]:
                            preds_all[k] = np.concatenate(preds_all[k], axis=0)
                        else:
                            preds_all[k] = np.array([])
                    for k in trues_all:
                        if trues_all[k]:
                            trues_all[k] = np.concatenate(trues_all[k], axis=0)
                        else:
                            trues_all[k] = np.array([])

                    # R2 metrics
                    r2s = {}
                    pairs = [
                        ('qrtend', 'qrtend_TAU', 'Rain Tendency (qrtend)'),
                        ('nctend', 'nctend_TAU', 'Cloud Number Tendency (nctend)'),
                        ('nrtend', 'nrtend_TAU', 'Rain Number Tendency (nrtend)'),
                        ('qctend', 'qctend_TAU', 'Cloud Water Tendency (qctend)')
                    ]
                    for pkey, tkey, _ in pairs:
                        if preds_all[pkey].size and trues_all[tkey].size:
                            try:
                                r2s[pkey] = float(r2_score(trues_all[tkey], preds_all[pkey]))
                            except Exception:
                                r2s[pkey] = float('nan')
                        else:
                            r2s[pkey] = float('nan')

                    # Active counts and classification accuracy/confusion
                    true_active = int(trues_all['is_active'].sum()) if trues_all['is_active'].size else 0
                    # Predicted active: threshold 0.5 on predicted probability if available
                    if preds_all['is_active'].size:
                        pred_active = int((preds_all['is_active'] >= 0.5).sum())
                    else:
                        pred_active = 0

                    # Epoch-level classification accuracy and confusion matrix
                    if preds_all['is_active'].size and trues_all['is_active'].size:
                        y_true_bin = (trues_all['is_active'] >= 0.5).astype(int)
                        y_pred_bin = (preds_all['is_active'] >= 0.5).astype(int)
                        cls_acc = float((y_true_bin == y_pred_bin).mean())
                        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
                        tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
                        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
                        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
                    else:
                        cls_acc, tp, tn, fp, fn = float('nan'), 0, 0, 0, 0

                    # Log scalars
                    wandb.log({
                        'train/loss_total': train_metrics['train_loss'],
                        'train/loss_cls': train_metrics.get('classification_loss', train_metrics.get('train_classification_loss', 0.0)),
                        'train/loss_reg': train_metrics.get('regression_loss', train_metrics.get('train_regression_loss', 0.0)),
                        'val/loss_total': val_metrics['val_loss'],
                        'val/loss_cls': val_metrics.get('val_classification_loss', 0.0),
                        'val/loss_reg': val_metrics.get('val_regression_loss', 0.0),
                        'metrics/r2_qrtend': r2s['qrtend'],
                        'metrics/r2_nctend': r2s['nctend'],
                        'metrics/r2_nrtend': r2s['nrtend'],
                        'metrics/r2_qctend': r2s['qctend'],
                        'val/active_true': true_active,
                        'val/active_pred': pred_active,
                        'val/cls_accuracy': cls_acc,
                        'val/confusion_tp': tp,
                        'val/confusion_tn': tn,
                        'val/confusion_fp': fp,
                        'val/confusion_fn': fn,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch
                    }, step=epoch)

                    # Confusion matrix panel (best-effort)
                    try:
                        wandb.log({'val/confusion_matrix':
                                   wandb.plot.confusion_matrix(
                                       y_true=y_true_bin.tolist(),
                                       preds=y_pred_bin.tolist(),
                                       class_names=['inactive', 'active']
                                   )}, step=epoch)
                    except Exception:
                        pass

                    # Image logging cadence
                    if epoch % self.image_log_interval == 0:
                        # Create classification-aware scatter plots
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        axes = axes.flatten()
                        
                        # Determine correctly vs incorrectly classified points
                        if preds_all['is_active'].size and trues_all['is_active'].size:
                            y_true_bin = (trues_all['is_active'] >= 0.5).astype(bool)
                            y_pred_bin = (preds_all['is_active'] >= 0.5).astype(bool)
                            correct_mask = (y_true_bin == y_pred_bin)
                            incorrect_mask = ~correct_mask
                            
                            # Count misclassified points for info
                            n_correct = correct_mask.sum()
                            n_incorrect = incorrect_mask.sum()
                            n_total = len(correct_mask)
                            misclass_rate = n_incorrect / n_total if n_total > 0 else 0.0
                        else:
                            # Fallback: treat all as correctly classified if no classification data
                            correct_mask = np.ones(len(preds_all[pairs[0][0]]), dtype=bool) if preds_all[pairs[0][0]].size else np.array([], dtype=bool)
                            incorrect_mask = np.zeros_like(correct_mask, dtype=bool)
                            misclass_rate = 0.0
                            n_correct = len(correct_mask)
                            n_incorrect = 0
                        
                        for ax, (pkey, tkey, title) in zip(axes, pairs):
                            y_pred = preds_all[pkey]
                            y_true = trues_all[tkey]
                            if y_pred.size and y_true.size:
                                # Plot correctly classified points as blue dots
                                if correct_mask.sum() > 0:
                                    ax.scatter(y_true[correct_mask], y_pred[correct_mask], 
                                             s=2, alpha=0.3, color='blue', label='Correctly Classified')
                                
                                # Plot misclassified points as red crosses
                                if incorrect_mask.sum() > 0:
                                    ax.scatter(y_true[incorrect_mask], y_pred[incorrect_mask], 
                                             s=6, alpha=0.7, color='red', marker='x', 
                                             label='Misclassified', linewidth=1)
                                
                                # Perfect prediction line
                                lo = float(min(y_true.min(), y_pred.min()))
                                hi = float(max(y_true.max(), y_pred.max()))
                                ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5)
                                
                                r2 = r2s[pkey]
                                ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=11)
                                
                                # Add legend only to first subplot to avoid clutter
                                if ax == axes[0] and (correct_mask.sum() > 0 or incorrect_mask.sum() > 0):
                                    ax.legend(fontsize=8, loc='upper left')
                            else:
                                ax.set_title(f"{title}\n(no data)", fontsize=11)
                            ax.set_xlabel('True', fontsize=9)
                            ax.set_ylabel('Pred', fontsize=9)
                            ax.grid(True, alpha=0.2)
                        
                        # Add overall misclassification info to the figure
                        fig.suptitle(f'Tendency Predictions \n (Misclassification Rate: {misclass_rate:.2%})', 
                                   fontsize=14, y=0.98)
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.94)  # Make room for suptitle
                        wandb.log({'val/scatter_all': wandb.Image(fig)}, step=epoch)
                        plt.close(fig)
                        
                        # Create histogram plots for tendencies
                        hist_fig, hist_axes = plt.subplots(2, 2, figsize=(12, 10))
                        hist_axes = hist_axes.flatten()
                        for ax, (pkey, tkey, title) in zip(hist_axes, pairs):
                            y_pred = preds_all[pkey]
                            y_true = trues_all[tkey]
                            if y_pred.size and y_true.size:
                                # Determine common range for both histograms
                                min_val = min(y_true.min(), y_pred.min())
                                max_val = max(y_true.max(), y_pred.max())
                                bins = np.linspace(min_val, max_val, 50)
                                
                                # Plot overlaid histograms
                                true_counts, _, _ = ax.hist(
                                    y_true,
                                    bins=bins,
                                    alpha=0.7,
                                    label='True',
                                    color='blue',
                                    density=False
                                )
                                pred_counts, _, _ = ax.hist(
                                    y_pred,
                                    bins=bins,
                                    alpha=0.7,
                                    label='Predicted',
                                    color='red',
                                    density=False
                                )

                                # Apply log-scale to counts
                                positive_counts = np.concatenate(
                                    [true_counts[true_counts > 0], pred_counts[pred_counts > 0]]
                                )
                                if positive_counts.size:
                                    ax.set_ylim(bottom=positive_counts.min())
                                else:
                                    ax.set_ylim(bottom=1.0)
                                ax.set_yscale('log')
                                
                                ax.set_xlabel(f'{title} Values', fontsize=9)
                                ax.set_ylabel('Count', fontsize=9)
                                ax.set_title(f'{title} - Distribution Comparison', fontsize=11)
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                            else:
                                ax.set_title(f"{title}\n(no data)", fontsize=11)
                                ax.set_xlabel('Values', fontsize=9)
                                ax.set_ylabel('Count', fontsize=9)
                        plt.tight_layout()
                        wandb.log({'val/histograms_all': wandb.Image(hist_fig)}, step=epoch)
                        plt.close(hist_fig)
                except Exception as e:
                    logger.warning(f"W&B epoch-level logging failed: {e}")
            
            # Logging (only on main process)
            if self.is_main_process:
                epoch_time = time.time() - epoch_start_time
                print(
                    f"Epoch {epoch+1}/{self.epochs} Summary: "
                    f"Train Loss: {train_metrics['train_loss']:.6f} "
                    f"(CLS: {train_metrics['classification_loss']:.4f}, REG: {train_metrics['regression_loss']:.4f}), "
                    f"Val Loss: {val_metrics['val_loss']:.6f} "
                    f"(CLS: {val_metrics['val_classification_loss']:.4f}, REG: {val_metrics['val_regression_loss']:.4f}), "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # W&B logging
                if self.use_wandb:
                    wandb.log(all_metrics, step=epoch)
                
                # Store history
                self.train_losses.append(train_metrics['train_loss'])
                self.val_losses.append(val_metrics['val_loss'])
            
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
            
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
        config={'training': {}, 'data': {}, 'logging': {'log_frequency': 1}}, # Added log_frequency
        rank=rank,
        world_size=world_size
    )
    
    print(f"✅ Trainer device: {trainer.device}")
    print(f"✅ Distributed: {trainer.is_distributed}")
    print(f"✅ Mixed precision: {trainer.use_amp}")
    print(f"✅ Log frequency: {trainer.log_frequency}") # Added log_frequency check
    
    trainer.cleanup()
    print("✅ Parallel training setup test completed!") 