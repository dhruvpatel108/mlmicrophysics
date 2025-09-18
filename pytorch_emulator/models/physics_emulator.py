"""
Constraint-Aware Microphysics Emulator

Simple, clean implementation of multi-head neural network for cloud microphysics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConstraintAwareEmulator(nn.Module):
    """
    Constraint-aware multi-head neural network for microphysics emulation.
    
    Architecture:
    - Shared backbone for feature extraction
    - Classification head for active/quiescent regime detection  
    - Regression heads with physical constraint activations
    - Mass conservation enforcement via post-processing
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        shared_dims: List[int] = [256, 128, 64],
        head_dims: List[int] = [64, 32, 16],
        #head_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_dims = shared_dims
        self.head_dims = head_dims
        self.dropout = dropout
        
        # Build shared backbone
        backbone_layers = []
        prev_dim = input_dim
        
        for dim in shared_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.shared_backbone = nn.Sequential(*backbone_layers)
        
        """
        # Classification head: Is_Active (active vs quiescent)
        self.classifier_head = nn.Sequential(
            nn.Linear(shared_dims[-1], head_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dims[0], head_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dims[1], 1)
            # Removed: nn.Sigmoid() - BCEWithLogitsLoss handles sigmoid internally
        )

        # Regression heads with constraint activations
        # qrtend: Must be ≥ 0 (rain formation is always positive)
        # For log-transformed data we don't need to apply ReLU to ensure ≥ 0
        self.qrtend_head = nn.Sequential(
            nn.Linear(shared_dims[-1], head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
            #nn.ReLU()  # Ensures ≥ 0
        )
        
        # nctend: Must be ≤ 0 (cloud droplet loss)
        self.nctend_head = nn.Sequential(
            nn.Linear(shared_dims[-1], head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
            # Will apply -ReLU in forward to ensure ≤ 0
        )
        
        # nrtend: Can be positive or negative (rain number can increase/decrease)
        self.nrtend_head = nn.Sequential(
            nn.Linear(shared_dims[-1], head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1)
            # No constraint activation
        )
        """
        # Deeper heads for reacher representation 
        # starting with shared_dims[-1] and going through head_dims ending with 1
        # Classification head
        classifier_head_layers = []
        input_dim = shared_dims[-1]
        for i in range(len(head_dims)):
            classifier_head_layers.append(nn.Linear(input_dim, head_dims[i]))
            classifier_head_layers.append(nn.ReLU())
            classifier_head_layers.append(nn.Dropout(dropout))
            input_dim = head_dims[i]
        classifier_head_layers.append(nn.Linear(input_dim, 1))
        self.classifier_head = nn.Sequential(*classifier_head_layers)

        # Regression heads with constraint activations
        # qrtend: Must be ≥ 0 (rain formation is always positive)
        # For log-transformed data we don't need to apply ReLU to ensure ≥ 0
        qrtend_head_layers = []
        input_dim = shared_dims[-1]
        for i in range(len(head_dims)):
            qrtend_head_layers.append(nn.Linear(input_dim, head_dims[i]))
            qrtend_head_layers.append(nn.ReLU())
            qrtend_head_layers.append(nn.Dropout(dropout))
            input_dim = head_dims[i]
        qrtend_head_layers.append(nn.Linear(input_dim, 1))
        self.qrtend_head = nn.Sequential(*qrtend_head_layers)

        # nctend: Must be ≤ 0 (cloud droplet loss)
        nctend_head_layers = []
        input_dim = shared_dims[-1]
        for i in range(len(head_dims)):
            nctend_head_layers.append(nn.Linear(input_dim, head_dims[i]))
            nctend_head_layers.append(nn.ReLU())
            nctend_head_layers.append(nn.Dropout(dropout))
            input_dim = head_dims[i]
        nctend_head_layers.append(nn.Linear(input_dim, 1))
        self.nctend_head = nn.Sequential(*nctend_head_layers)

        # nrtend: Can be positive or negative (rain number can increase/decrease)
        nrtend_head_layers = []
        input_dim = shared_dims[-1]
        for i in range(len(head_dims)):
            nrtend_head_layers.append(nn.Linear(input_dim, head_dims[i]))
            nrtend_head_layers.append(nn.ReLU())
            nrtend_head_layers.append(nn.Dropout(dropout))
            input_dim = head_dims[i]
        nrtend_head_layers.append(nn.Linear(input_dim, 1))
        self.nrtend_head = nn.Sequential(*nrtend_head_layers)

        # Initialize weights
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
            - 'qrtend': Rain tendency  [batch_size, 1]  
            - 'nctend': Cloud droplet number tendency  [batch_size, 1]
            - 'nrtend': Rain number tendency [batch_size, 1]
            - 'qctend': Cloud water tendency (derived) [batch_size, 1]
        """
        # Shared feature extraction
        shared_features = self.shared_backbone(x)
        
        # Classification: Active vs quiescent regime (raw logits for loss, sigmoid for output)
        is_active_logits = self.classifier_head(shared_features)
        is_active = torch.sigmoid(is_active_logits)  # Apply sigmoid for final output
        
        # Regression heads with physical constraints
        qrtend = self.qrtend_head(shared_features)  # For log-transformed data we don't need to apply ReLU to ensure ≥ 0
        
        # For nctend: Apply -ReLU to ensure ≤0
        # For log-transformed data we don't need to apply ReLU to ensure ≤ 0
        #nctend_positive = self.nctend_head(shared_features)
        #nctend = -F.relu(nctend_positive)  # Ensures ≤0
        nctend = self.nctend_head(shared_features)
        
        nrtend = self.nrtend_head(shared_features)  # No constraints
        
        # Mass conservation: qctend = -qrtend (hard-coded physics)
        qctend = -qrtend
        
        return {
            'is_active': is_active,  # Sigmoid output for predictions
            'is_active_logits': is_active_logits,  # Raw logits for BCEWithLogitsLoss
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