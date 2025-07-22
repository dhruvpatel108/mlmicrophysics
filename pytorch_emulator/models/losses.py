"""
Constraint-Aware Loss Functions

Custom loss functions for the constraint-aware microphysics emulator,
combining classification and regression objectives with physical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConstraintAwareLoss(nn.Module):
    """
    Combined loss function for constraint-aware microphysics emulation.
    
    Combines:
    - Classification loss (BCE) for active/quiescent detection
    - Regression loss (Huber) for tendency predictions (only on active samples)
    - Mass conservation penalty (optional)
    
    Args:
        alpha: Weight balance between classification and regression loss
        huber_delta: Delta parameter for Huber loss
        conservation_weight: Weight for mass conservation penalty
        use_masking: Whether to mask regression loss to active samples only
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        huber_delta: float = 1.0,
        conservation_weight: float = 0.1,
        use_masking: bool = True
    ):
        super().__init__()
        
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.conservation_weight = conservation_weight
        self.use_masking = use_masking
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()  # Changed from BCELoss for autocast compatibility
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined constraint-aware loss.
        
        Args:
            predictions: Model output dictionary with keys:
                - 'is_active': [batch_size, 1] classification probabilities
                - 'is_active_logits': [batch_size, 1] raw logits for loss computation
                - 'qrtend': [batch_size, 1] rain tendency predictions (≥0)
                - 'nctend': [batch_size, 1] cloud number tendency predictions (≤0)
                - 'nrtend': [batch_size, 1] rain number tendency predictions
                - 'qctend': [batch_size, 1] derived cloud water tendency (-qrtend)
                
            targets: Ground truth dictionary with keys:
                - 'is_active': [batch_size, 1] binary labels (0/1)
                - 'qctend_TAU': [batch_size, 1] target cloud water tendencies
                - 'nctend_TAU': [batch_size, 1] target cloud number tendencies
                - 'nrtend_TAU': [batch_size, 1] target rain number tendencies
                - 'qrtend_TAU': [batch_size, 1] target rain water tendencies
                
        Returns:
            Dictionary containing individual and total losses
        """
        batch_size = predictions['is_active'].shape[0]
        
        # 1. Classification Loss (Binary Cross-Entropy with Logits)
        classification_loss = self.bce_loss(
            predictions['is_active_logits'],  # Use raw logits for BCEWithLogitsLoss
            targets['is_active']
        )
        
        # 2. Regression Loss (Huber Loss)
        if self.use_masking:
            # Only compute regression loss on active samples
            active_mask = targets['is_active'].bool().squeeze()
            
            if active_mask.sum() > 0:  # If any active samples exist
                # Compute regression loss only on active samples
                qr_loss = self.huber_loss(
                    predictions['qrtend'][active_mask], 
                    targets['qrtend_TAU'][active_mask]
                )
                nc_loss = self.huber_loss(
                    predictions['nctend'][active_mask], 
                    targets['nctend_TAU'][active_mask]
                )
                nr_loss = self.huber_loss(
                    predictions['nrtend'][active_mask], 
                    targets['nrtend_TAU'][active_mask]
                )
                
                regression_loss = (qr_loss + nc_loss + nr_loss) / 3.0
            else:
                # No active samples - zero regression loss
                regression_loss = torch.tensor(0.0, device=predictions['is_active'].device)
        else:
            # Compute regression loss on all samples
            qr_loss = self.huber_loss(predictions['qrtend'], targets['qrtend_TAU'])
            nc_loss = self.huber_loss(predictions['nctend'], targets['nctend_TAU'])
            nr_loss = self.huber_loss(predictions['nrtend'], targets['nrtend_TAU'])
            
            regression_loss = (qr_loss + nc_loss + nr_loss) / 3.0
        
        # 3. Combined Loss (no conservation penalty needed since qctend = -qrtend is hardcoded)
        total_loss = (
            self.alpha * classification_loss + 
            (1 - self.alpha) * regression_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'active_samples': active_mask.sum().item() if self.use_masking else batch_size
        }
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weighting configuration."""
        return {
            'alpha': self.alpha,
            'classification_weight': self.alpha,
            'regression_weight': 1 - self.alpha,
            'conservation_weight': self.conservation_weight,
            'huber_delta': self.huber_delta
        }


def create_constraint_aware_loss(config: Dict) -> ConstraintAwareLoss:
    """
    Factory function to create ConstraintAwareLoss from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized ConstraintAwareLoss
    """
    loss_config = config.get('model', {})
    
    return ConstraintAwareLoss(
        alpha=loss_config.get('alpha', 0.3),
        huber_delta=loss_config.get('huber_delta', 1.0),
        conservation_weight=loss_config.get('conservation_weight', 0.1),
        use_masking=loss_config.get('use_masking', True)
    )


# Test the loss function
if __name__ == "__main__":
    print("Testing ConstraintAwareLoss...")
    
    # Create loss function
    loss_fn = ConstraintAwareLoss(alpha=0.3, huber_delta=1.0)
    
    # Create sample predictions and targets
    batch_size = 16
    
    predictions = {
        'is_active': torch.sigmoid(torch.randn(batch_size, 1)),
        'is_active_logits': torch.randn(batch_size, 1), # Added is_active_logits
        'qrtend': torch.relu(torch.randn(batch_size, 1)),  # ≥ 0
        'nctend': -torch.relu(torch.randn(batch_size, 1)), # ≤ 0
        'nrtend': torch.randn(batch_size, 1),              # Any value
    }
    # Add derived qctend
    predictions['qctend'] = -predictions['qrtend']
    
    targets = {
        'is_active': torch.randint(0, 2, (batch_size, 1)).float(),
        'qctend_TAU': torch.abs(torch.randn(batch_size, 1)),
        'nctend_TAU': -torch.abs(torch.randn(batch_size, 1)),
        'nrtend_TAU': torch.randn(batch_size, 1),
        'qrtend_TAU': torch.abs(torch.randn(batch_size, 1))
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets)
    
    print(f"✅ Loss computation successful!")
    print(f"  Total loss: {losses['total_loss'].item():.6f}")
    print(f"  Classification: {losses['classification_loss'].item():.6f}")
    print(f"  Regression: {losses['regression_loss'].item():.6f}")
    print(f"  Active samples: {losses['active_samples']}/{batch_size}")
    
    print(f"\nLoss weights: {loss_fn.get_loss_weights()}") 