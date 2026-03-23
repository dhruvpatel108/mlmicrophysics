"""
Mixture-of-Experts Constraint-Aware Microphysics Emulator

Replaces the single nrtend head with a 3-expert MoE head to improve
prediction across three physical regimes:
  Expert 0: near-zero  (|nrtend| < eps)  -- negligible rain-drop change
  Expert 1: negative   (nrtend < -eps)   -- self-collection
  Expert 2: positive   (nrtend > eps)    -- rain formation (accretion/autoconversion)

Training uses soft routing (weighted sum of expert outputs).
Inference uses hard routing (argmax selects a single expert).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.physics_emulator import _get_activation, build_head

logger = logging.getLogger(__name__)

# Regime indices (match the labels produced by the data loader)
REGIME_NEAR_ZERO = 0
REGIME_NEGATIVE = 1  # self-collection
REGIME_POSITIVE = 2  # rain formation
N_REGIMES = 3


class MoENrtendHead(nn.Module):
    """3-expert MoE head for nrtend with a multi-layer router.

    Experts:
        Each expert is a small MLP that produces a scalar nrtend prediction.
        Expert 0 -- near-zero regime
        Expert 1 -- negative (self-collection)
        Expert 2 -- positive (rain formation)

    Router:
        Multi-layer MLP that produces 3-class logits used as soft gate
        weights during training and hard (argmax) selection at inference.
    """

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dims: List[int] = [128, 64, 32],
        router_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()

        # --- Experts ---
        self.experts = nn.ModuleList([
            build_head(input_dim, expert_hidden_dims, output_dim=1,
                       activation=activation, dropout=dropout)
            for _ in range(N_REGIMES)
        ])

        # --- Router (multi-layer MLP) ---
        self.router = build_head(
            input_dim, router_hidden_dims, output_dim=N_REGIMES,
            activation=activation, dropout=dropout,
        )

    def forward(self, shared_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shared_features: [B, input_dim] from shared backbone.

        Returns:
            nrtend:       [B, 1]          blended (train) or selected (eval) prediction.
            router_logits: [B, N_REGIMES]  raw logits for gating loss / diagnostics.
        """
        router_logits = self.router(shared_features)  # [B, 3]

        expert_outputs = torch.cat(
            [expert(shared_features) for expert in self.experts], dim=-1
        )  # [B, 3]

        if self.training:
            # Soft routing: weighted sum of all experts
            gate_probs = F.softmax(router_logits, dim=-1)  # [B, 3]
            nrtend = (gate_probs * expert_outputs).sum(dim=-1, keepdim=True)
        else:
            # Hard routing: pick expert with highest gate probability
            regime = router_logits.argmax(dim=-1)  # [B]
            nrtend = expert_outputs.gather(
                1, regime.unsqueeze(-1)
            )  # [B, 1]

        return nrtend, router_logits


class MoEConstraintAwareEmulator(nn.Module):
    """Constraint-aware emulator with a 3-expert MoE head for nrtend.

    Architecture
    ────────────
    - Shared backbone  (ReLU, loaded from pretrained ConstraintAwareEmulator)
    - qrtend_head      (ReLU, loaded from pretrained)
    - nctend_head      (ReLU, loaded from pretrained)
    - MoE nrtend head  (SiLU, randomly initialised)
    - qctend = -qrtend (mass conservation)

    No classification head.
    """

    def __init__(
        self,
        input_dim: int = 11,
        shared_dims: List[int] = [256, 256, 256, 128, 128],
        head_dims: List[int] = [128, 128, 64, 64, 32],
        dropout: float = 0.0,
        activation: str = "relu",
        # MoE-specific
        n_experts: int = N_REGIMES,
        expert_hidden_dims: Optional[List[int]] = None,
        router_hidden_dims: Optional[List[int]] = None,
        moe_activation: str = "silu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.shared_dims = shared_dims
        self.head_dims = head_dims
        self.dropout = dropout
        self.activation_name = (activation or "relu").lower()
        self.moe_activation_name = (moe_activation or "silu").lower()
        self.n_experts = n_experts

        # ── Shared backbone (identical to ConstraintAwareEmulator) ──
        backbone_layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in shared_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, dim),
                _get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.shared_backbone = nn.Sequential(*backbone_layers)

        backbone_out = shared_dims[-1]

        # ── Standard regression heads (ReLU, same as pretrained) ──
        self.qrtend_head = build_head(backbone_out, head_dims, 1, activation, dropout)
        self.nctend_head = build_head(backbone_out, head_dims, 1, activation, dropout)

        # ── MoE head for nrtend (SiLU, randomly initialised) ──
        if expert_hidden_dims is None:
            expert_hidden_dims = [128, 64, 32]
        if router_hidden_dims is None:
            router_hidden_dims = [64, 32]

        self.nrtend_moe = MoENrtendHead(
            input_dim=backbone_out,
            expert_hidden_dims=expert_hidden_dims,
            router_hidden_dims=router_hidden_dims,
            dropout=dropout,
            activation=moe_activation,
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    # ── Pretrained weight loading ──────────────────────────────────

    def load_pretrained_backbone_and_heads(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load backbone + qrtend/nctend weights from a ConstraintAwareEmulator checkpoint.

        Loads:  shared_backbone, qrtend_head, nctend_head
        Skips:  classifier_head, nrtend_head (not present in MoE model)

        The MoE nrtend head keeps its random initialisation.
        """
        map_location = device or "cpu"
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        src_state = ckpt.get("model_state_dict", ckpt)

        loaded, skipped = [], []
        own_state = self.state_dict()

        for name, param in src_state.items():
            if name.startswith(("classifier_head.", "nrtend_head.")):
                skipped.append(name)
                continue
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)

        self.load_state_dict(own_state, strict=False)
        logger.info(
            "Loaded %d pretrained params (backbone + qrtend + nctend). "
            "Skipped %d (classifier/nrtend/shape-mismatch).",
            len(loaded), len(skipped),
        )

    # ── Freeze / unfreeze helpers for phased training ──────────────
    #
    # Phase 1 (freeze_pretrained=true):
    #   Only the MoE nrtend head (router + experts) is trainable.
    #   Backbone and qrtend/nctend heads are frozen.
    #
    # Phase 2 (freeze_pretrained=false):
    #   Everything is unfrozen, use a small LR (e.g. 1e-4).
    #   To switch: set  model.moe.freeze_pretrained: false  in the config
    #   and submit a new run (or manually call model.unfreeze_all()).

    def freeze_backbone_and_standard_heads(self) -> None:
        """Freeze shared_backbone, qrtend_head, nctend_head (Phase 1)."""
        for name, param in self.named_parameters():
            if name.startswith(("shared_backbone.", "qrtend_head.", "nctend_head.")):
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter (Phase 2)."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_backbone(x)

        qrtend = self.qrtend_head(shared_features)
        nctend = self.nctend_head(shared_features)
        nrtend, router_logits = self.nrtend_moe(shared_features)
        qctend = -qrtend

        return {
            "qrtend": qrtend,
            "nctend": nctend,
            "nrtend": nrtend,
            "qctend": qctend,
            "router_logits": router_logits,
        }

    # ── Utilities ──────────────────────────────────────────────────

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = self.get_parameter_count()
        return {
            "architecture": "moe",
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "input_dim": self.input_dim,
            "shared_dims": self.shared_dims,
            "head_dims": self.head_dims,
            "n_experts": self.n_experts,
            "backbone_activation": self.activation_name,
            "moe_activation": self.moe_activation_name,
            "dropout": self.dropout,
            "constraints": {
                "qrtend": "unconstrained (log-space)",
                "nctend": "unconstrained (log-space)",
                "nrtend": "MoE 3-expert head",
                "mass_conservation": "qctend = -qrtend",
            },
        }


# ── Quick self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    model = MoEConstraintAwareEmulator(
        shared_dims=[256, 256, 256, 128, 128],
        head_dims=[128, 128, 64, 64, 32],
        expert_hidden_dims=[128, 64, 32],
        router_hidden_dims=[64, 32],
        moe_activation="silu",
    )
    print(f"MoE model: {model.get_parameter_count():,} trainable parameters")
    print(model.get_model_info())

    x = torch.randn(16, 11)

    # Training mode (soft routing)
    model.train()
    out_train = model(x)
    print("\n[train] Output shapes:")
    for k, v in out_train.items():
        print(f"  {k}: {v.shape}")

    # Eval mode (hard routing)
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print("\n[eval] Output shapes:")
    for k, v in out_eval.items():
        print(f"  {k}: {v.shape}")

    print(f"\nMass conservation: {torch.allclose(out_eval['qctend'], -out_eval['qrtend'])}")

    # Test freeze/unfreeze
    model.freeze_backbone_and_standard_heads()
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 1 (frozen): {trainable_p1:,} / {total:,} trainable")

    model.unfreeze_all()
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2 (unfrozen): {trainable_p2:,} / {total:,} trainable")
