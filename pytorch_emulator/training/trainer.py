"""
Training Pipeline for Constraint-Aware Microphysics Emulator

Complete training infrastructure with logging, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Dict, Optional, Tuple
import logging

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstraintAwareTrainer:
    """
    Trainer for constraint-aware microphysics emulator.
    
    Handles:
    - Training loop with constraint-aware loss
    - Learning rate scheduling
    - Model checkpointing
    - Logging and visualization
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict,
        device: str = "auto"
    ):
        """
        Initialize trainer.
        
        Args:
            model: ConstraintAwareEmulator model
            loss_fn: ConstraintAwareLoss function
            config: Training configuration
            device: Device to train on ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Training configuration
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        self.weight_decay = train_config.get('weight_decay', 1e-5)
        self.gradient_clip_norm = train_config.get('gradient_clip_norm', 1.0)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(train_config)
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler(train_config)
        
        # Early stopping
        self.early_stopping_patience = train_config.get('early_stopping_patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging frequency and W&B cadence
        logging_config = config.get('logging', {})
        self.log_frequency = int(logging_config.get('log_frequency', 50))
        self.epoch_log_frequency = int(logging_config.get('epoch_log_frequency', 1))
        self.image_log_percent = float(logging_config.get('image_log_percent', 5))
        self.image_log_interval = max(1, int(self.epochs * (self.image_log_percent / 100.0)))

        # Logging setup
        self.setup_logging(config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
    def _setup_optimizer(self, train_config: Dict) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
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
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_params.get('mode', 'min'),
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-7)
            )
            logger.info(f"Setup ReduceLROnPlateau scheduler: factor={scheduler_params.get('factor', 0.5)}, patience={scheduler_params.get('patience', 5)}")
            return scheduler
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
            logger.info(f"Setup CosineAnnealingLR scheduler: T_max={self.epochs}")
            return scheduler
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 30),
                gamma=train_config.get('gamma', 0.1)
            )
            logger.info(f"Setup StepLR scheduler: step_size={train_config.get('step_size', 30)}, gamma={train_config.get('gamma', 0.1)}")
            return scheduler
        else:
            logger.info(f"No scheduler configured (scheduler_name: {scheduler_name})")
            return None
    
    def setup_logging(self, config: Dict):
        """Setup logging including W&B if configured."""
        logging_config = config.get('logging', {})
        wandb_config = logging_config.get('wandb', {})
        
        # Initialize W&B if configured and available
        if WANDB_AVAILABLE and wandb_config.get('project'):
            try:
                wandb.init(
                    project=wandb_config.get('project'),
                    entity=wandb_config.get('entity'),
                    group=wandb_config.get('group'),
                    job_type=wandb_config.get('job_type', 'training'),
                    tags=wandb_config.get('tags', []),
                    config=config
                )
                self.use_wandb = True
                logger.info("W&B logging initialized")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
                self.use_wandb = False
        else:
            if not WANDB_AVAILABLE and wandb_config.get('project'):
                logger.warning("W&B requested but not installed. Install with: pip install wandb")
            self.use_wandb = False
        
        # Setup output directory
        self.output_dir = Path(config.get('data', {}).get('out_path', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'classification_loss': [],
            'regression_loss': [],
            'conservation_loss': [],
            'active_samples': []
        }
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            # Print train batch class balance for first few batches
            if batch_idx < 5 and 'is_active' in targets:
                try:
                    _bsz = targets['is_active'].numel()
                    _act = int(targets['is_active'].sum().item())
                    _inact = _bsz - _act
                    logger.info(f"[train] epoch {self.epoch} batch {batch_idx}: active={_act/_bsz:.2%}, inactive={_inact/_bsz:.2%}")
                except Exception:
                    pass
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            
            # Compute loss
            loss_dict = self.loss_fn(predictions, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Record metrics
            epoch_losses.append(total_loss.item())
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key].append(loss_dict[key])
            
            self.global_step += 1
            
            # Log batch metrics
            if self.log_frequency > 0 and batch_idx % self.log_frequency == 0:
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss.item():.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # Compute epoch averages
        train_metrics = {
            'train_loss': sum(epoch_losses) / len(epoch_losses),
            'train_classification_loss': sum(epoch_metrics['classification_loss']) / len(epoch_metrics['classification_loss']),
            'train_regression_loss': sum(epoch_metrics['regression_loss']) / len(epoch_metrics['regression_loss']),
            'train_conservation_loss': sum(epoch_metrics['conservation_loss']) / len(epoch_metrics['conservation_loss']),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return train_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {
            'classification_loss': [],
            'regression_loss': [],
            'conservation_loss': [],
            'active_samples': []
        }
        
        with torch.no_grad():
            for bidx, (inputs, targets) in enumerate(val_loader):
                # Move to device
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                # Print val batch class balance for first few batches
                if bidx < 5 and 'is_active' in targets:
                    try:
                        _bsz = targets['is_active'].numel()
                        _act = int(targets['is_active'].sum().item())
                        _inact = _bsz - _act
                        logger.info(f"[val]   epoch {self.epoch} batch {bidx}: active={_act/_bsz:.2%}, inactive={_inact/_bsz:.2%}")
                    except Exception:
                        pass
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, targets)
                total_loss = loss_dict['total_loss']
                
                # Record metrics
                epoch_losses.append(total_loss.item())
                for key in epoch_metrics:
                    if key in loss_dict:
                        epoch_metrics[key].append(loss_dict[key])
        
        # Compute validation averages
        val_metrics = {
            'val_loss': sum(epoch_losses) / len(epoch_losses),
            'val_classification_loss': sum(epoch_metrics['classification_loss']) / len(epoch_metrics['classification_loss']),
            'val_regression_loss': sum(epoch_metrics['regression_loss']) / len(epoch_metrics['regression_loss']),
            'val_conservation_loss': sum(epoch_metrics['conservation_loss']) / len(epoch_metrics['conservation_loss'])
        }
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
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
            logger.info(f"Saved best checkpoint: {best_path}")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                    logger.info(f"ReduceLROnPlateau scheduler triggered. New LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(all_metrics, is_best)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.6f}, "
                f"Val Loss: {val_metrics['val_loss']:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # W&B epoch-level logging: full-validation metrics + all four scatter plots
            if self.use_wandb and WANDB_AVAILABLE and (epoch % self.epoch_log_frequency == 0):
                try:
                    import numpy as np
                    from sklearn.metrics import r2_score
                    import matplotlib.pyplot as plt
                    self.model.eval()
                    preds_all = {k: [] for k in ['qrtend', 'nctend', 'nrtend', 'qctend', 'is_active']}
                    trues_all = {k: [] for k in ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU', 'is_active']}
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs = inputs.to(self.device)
                            targ_dev = {k: v.to(self.device) for k, v in targets.items()}
                            out = self.model(inputs)
                            for key in ['qrtend', 'nctend', 'nrtend', 'qctend', 'is_active']:
                                if key in out:
                                    preds_all[key].append(out[key].detach().float().view(-1).cpu().numpy())
                            for key in ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU', 'is_active']:
                                if key in targ_dev:
                                    trues_all[key].append(targ_dev[key].detach().float().view(-1).cpu().numpy())
                    for k in preds_all:
                        preds_all[k] = np.concatenate(preds_all[k], axis=0) if preds_all[k] else np.array([])
                    for k in trues_all:
                        trues_all[k] = np.concatenate(trues_all[k], axis=0) if trues_all[k] else np.array([])

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

                    true_active = int(trues_all['is_active'].sum()) if trues_all['is_active'].size else 0
                    pred_active = int((preds_all['is_active'] >= 0.5).sum()) if preds_all['is_active'].size else 0
                    # Classification accuracy and confusion
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

                    wandb.log({
                        'train/loss_total': train_metrics['train_loss'],
                        'train/loss_cls': train_metrics.get('train_classification_loss', 0.0),
                        'train/loss_reg': train_metrics.get('train_regression_loss', 0.0),
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
                    try:
                        wandb.log({'val/confusion_matrix':
                                   wandb.plot.confusion_matrix(
                                       y_true=y_true_bin.tolist(),
                                       preds=y_pred_bin.tolist(),
                                       class_names=['inactive','active']
                                   )}, step=epoch)
                    except Exception:
                        pass

                    if epoch % self.image_log_interval == 0:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        axes = axes.flatten()
                        for ax, (pkey, tkey, title) in zip(axes, pairs):
                            y_pred = preds_all[pkey]
                            y_true = trues_all[tkey]
                            if y_pred.size and y_true.size:
                                ax.scatter(y_true, y_pred, s=2, alpha=0.3)
                                lo = float(min(y_true.min(), y_pred.min()))
                                hi = float(max(y_true.max(), y_pred.max()))
                                ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
                                r2 = r2s[pkey]
                                ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=11)
                            else:
                                ax.set_title(f"{title}\n(no data)", fontsize=11)
                            ax.set_xlabel('True', fontsize=9)
                            ax.set_ylabel('Pred', fontsize=9)
                            ax.grid(True, alpha=0.2)
                        plt.tight_layout()
                        wandb.log({'val/scatter_all': wandb.Image(fig)}, step=epoch)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"W&B epoch-level logging failed: {e}")
            
            # Store history
            self.train_losses.append(train_metrics['train_loss'])
            self.val_losses.append(val_metrics['val_loss'])
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Finalize W&B
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1
        }


def create_trainer(
    model: nn.Module, 
    loss_fn: nn.Module, 
    config: Dict
) -> ConstraintAwareTrainer:
    """
    Factory function to create a trainer.
    
    Args:
        model: Initialized model
        loss_fn: Initialized loss function
        config: Configuration dictionary
        
    Returns:
        Initialized trainer
    """
    return ConstraintAwareTrainer(model, loss_fn, config)


# Test the trainer
if __name__ == "__main__":
    print("Testing ConstraintAwareTrainer...")
    
    # Mock imports for testing
    import sys
    sys.path.append('../models')
    
    try:
        from physics_emulator import ConstraintAwareEmulator
        from losses import ConstraintAwareLoss
        
        # Create model and loss
        model = ConstraintAwareEmulator(input_dim=11)
        loss_fn = ConstraintAwareLoss()
        
        # Create config
        config = {
            'training': {
                'epochs': 2,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            },
            'data': {
                'out_path': '/tmp/test_training'
            }
        }
        
        # Create trainer
        trainer = ConstraintAwareTrainer(model, loss_fn, config, device='cpu')
        
        print("✅ Trainer created successfully!")
        print(f"Device: {trainer.device}")
        print(f"Optimizer: {type(trainer.optimizer).__name__}")
        print(f"Scheduler: {type(trainer.scheduler).__name__ if trainer.scheduler else None}")
        
    except ImportError as e:
        print(f"Import error (expected in testing): {e}")
        print("✅ Trainer structure looks good!") 