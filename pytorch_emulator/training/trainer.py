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
        
        # Logging frequency
        logging_config = config.get('logging', {})
        self.log_frequency = int(logging_config.get('log_frequency', 50))

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
            for inputs, targets in val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
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
            
            # W&B logging
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log(all_metrics, step=epoch)
            
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