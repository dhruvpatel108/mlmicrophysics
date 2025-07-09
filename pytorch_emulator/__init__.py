"""
PyTorch-based Constraint-Aware Microphysics Emulator

This package contains PyTorch implementations of neural network emulators
for cloud microphysics processes, designed to replace expensive microphysics
calculations in Earth System Models.
"""

__version__ = "1.0.0"

# Main exports
from .models.physics_emulator import ConstraintAwareEmulator
from .models.losses import ConstraintAwareLoss
from .training.trainer import ConstraintAwareTrainer

__all__ = [
    "ConstraintAwareEmulator",
    "ConstraintAwareLoss", 
    "ConstraintAwareTrainer"
] 