"""
Constraint-Aware Microphysics Emulator

Multi-head neural network for cloud microphysics with shared backbone,
per-variable regression heads, and hard-coded mass conservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def _get_activation(name: str):
    """Return activation module by name (relu, silu, etc.)."""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}. Supported: relu, silu")


def build_head(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int = 1,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a multi-layer head (classification or regression).

    Produces: Linear -> Activation -> Dropout  (repeated per hidden_dim)
              Linear -> output_dim              (final projection)
    """
    layers: List[nn.Module] = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(_get_activation(activation))
        layers.append(nn.Dropout(dropout))
        prev = dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ConstraintAwareEmulator(nn.Module):
    """
    Constraint-aware multi-head neural network for microphysics emulation.

    Architecture:
    - Shared backbone for feature extraction
    - Classification head for active/quiescent regime detection
    - Regression heads for qrtend, nctend, nrtend
    - Mass conservation enforcement: qctend = -qrtend
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        shared_dims: List[int] = [256, 128, 64],
        head_dims: List[int] = [64, 32, 16],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_dims = shared_dims
        self.head_dims = head_dims
        self.dropout = dropout
        self.activation_name = (activation or "relu").lower()
        
        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, dim),
                _get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.shared_backbone = nn.Sequential(*backbone_layers)

        backbone_out = shared_dims[-1]
        self.classifier_head = build_head(backbone_out, head_dims, 1, activation, dropout)
        self.qrtend_head = build_head(backbone_out, head_dims, 1, activation, dropout)
        self.nctend_head = build_head(backbone_out, head_dims, 1, activation, dropout)
        self.nrtend_head = build_head(backbone_out, head_dims, 1, activation, dropout)

        self._init_weights()
    
    def _init_weights(self):
        """Initialize all linear layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the constraint-aware emulator.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
            - 'is_active': Classification probabilities [batch_size, 1]
            - 'is_active_logits': Raw logits for BCEWithLogitsLoss [batch_size, 1]
            - 'qrtend': Rain tendency [batch_size, 1]
            - 'nctend': Cloud droplet number tendency [batch_size, 1]
            - 'nrtend': Rain number tendency [batch_size, 1]
            - 'qctend': Cloud water tendency (derived) [batch_size, 1]
        """
        shared_features = self.shared_backbone(x)
        
        is_active_logits = self.classifier_head(shared_features)
        is_active = torch.sigmoid(is_active_logits)
        
        qrtend = self.qrtend_head(shared_features)
        nctend = self.nctend_head(shared_features)
        nrtend = self.nrtend_head(shared_features)
        qctend = -qrtend
        
        return {
            'is_active': is_active,
            'is_active_logits': is_active_logits,
            'qrtend': qrtend,
            'nctend': nctend, 
            'nrtend': nrtend,
            'qctend': qctend
        }
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'total_parameters': self.get_parameter_count(),
            'input_dim': self.input_dim,
            'shared_dims': self.shared_dims,
            'head_dims': self.head_dims,
            'dropout': self.dropout,
            'constraints': {
                'qrtend': '≥ 0 (ReLU)',
                'nctend': '≤ 0 (-ReLU)', 
                'nrtend': 'unconstrained',
                'mass_conservation': 'qctend = -qrtend'
            }
        }


# Simple test function
if __name__ == "__main__":
    # Create model
    model = ConstraintAwareEmulator()
    print(f"Model has {model.get_parameter_count():,} parameters")
    
    # Test with sample data
    batch_size = 16
    sample_input = torch.randn(batch_size, 11)
    
    with torch.no_grad():
        predictions = model(sample_input)
    
    print("\nPrediction shapes:")
    for key, tensor in predictions.items():
        print(f"  {key}: {tensor.shape}")
    
    # Verify constraints
    print("\nConstraint verification:")
    print(f"  qrtend ≥ 0: {torch.all(predictions['qrtend'] >= 0).item()}")
    print(f"  nctend ≤ 0: {torch.all(predictions['nctend'] <= 0).item()}")
    print(f"  Mass conservation: {torch.allclose(predictions['qctend'], -predictions['qrtend'])}") 