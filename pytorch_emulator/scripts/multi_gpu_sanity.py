#!/usr/bin/env python3
"""
Synthetic multi-GPU smoke tests for the DataParallelTrainer.

Runs a tiny constraint-aware model on random data to verify:
  1. Single-GPU baseline behaviour
  2. DataParallel (single-process, multi-GPU) execution
  3. DistributedDataParallel (multi-process) execution via torchrun

Usage examples:
  # Single GPU baseline
  python scripts/multi_gpu_sanity.py

  # DataParallel on all visible GPUs
  CUDA_VISIBLE_DEVICES=0,1 python scripts/multi_gpu_sanity.py --data-parallel

  # DistributedDataParallel across 2 GPUs
  torchrun --nproc_per_node=2 scripts/multi_gpu_sanity.py --distributed
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from training.parallel_trainer import DataParallelTrainer, setup_distributed_training
from models.losses import ConstraintAwareLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU sanity checks with synthetic data")
    parser.add_argument("--distributed", action="store_true", help="Use DistributedDataParallel via torchrun")
    parser.add_argument("--data-parallel", action="store_true", help="Use nn.DataParallel within a single process")
    parser.add_argument("--num-samples", type=int, default=4096, help="Number of synthetic training samples")
    parser.add_argument("--val-samples", type=int, default=512, help="Number of synthetic validation samples")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to train")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_multi_gpu_outputs",
        help="Directory to store checkpoints and result summaries",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default="synthetic_results.json",
        help="Filename for JSON metrics summary (written by rank 0)",
    )
    return parser.parse_args()


class SyntheticMicrophysicsDataset(Dataset):
    """Simple random dataset mimicking the emulator batch structure."""

    def __init__(self, num_samples: int, input_dim: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(num_samples, input_dim, generator=g)

        weights = torch.randn(input_dim, 4, generator=g)
        bias = torch.randn(4, generator=g)
        raw_targets = self.inputs @ weights + bias

        # Construct targets matching ConstraintAwareLoss expectations
        self.targets = {
            "is_active": torch.sigmoid(raw_targets[:, 0:1]).ge(0.5).float(),
            "qrtend_TAU": torch.relu(raw_targets[:, 1:2]),
            "nctend_TAU": -torch.relu(raw_targets[:, 2:3]),
            "nrtend_TAU": raw_targets[:, 3:4],
        }
        self.targets["qctend_TAU"] = -self.targets["qrtend_TAU"]

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_targets = {k: v[idx].clone() for k, v in self.targets.items()}
        return self.inputs[idx].clone(), sample_targets


class SyntheticConstraintModel(nn.Module):
    """Tiny MLP that outputs the dictionary expected by ConstraintAwareLoss."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.regression = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        logits = self.classifier(h)
        reg = self.regression(h)

        qrtend = torch.relu(reg[:, 0:1])
        nctend = -torch.relu(reg[:, 1:2])
        nrtend = reg[:, 2:3]
        qctend = -qrtend

        return {
            "is_active_logits": logits,
            "is_active": torch.sigmoid(logits),
            "qrtend": qrtend,
            "nctend": nctend,
            "nrtend": nrtend,
            "qctend": qctend,
        }


def build_dataloaders(args: argparse.Namespace, rank: int, world_size: int):
    train_dataset = SyntheticMicrophysicsDataset(
        num_samples=args.num_samples,
        seed=args.seed + rank,
    )
    val_dataset = SyntheticMicrophysicsDataset(
        num_samples=args.val_samples,
        seed=args.seed + 10 + rank,
    )

    use_distributed = args.distributed and world_size > 1

    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        if use_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed + 1
        )
        if use_distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def vector_hash(model: nn.Module) -> str:
    device = next(model.parameters()).device
    with torch.no_grad():
        vec = parameters_to_vector(model.parameters()).detach().to(device)
        vec_cpu = vec.float().cpu().numpy().tobytes()
    return hashlib.sha256(vec_cpu).hexdigest()


def main():
    args = parse_args()

    if args.distributed and args.data_parallel:
        raise ValueError("Use either --distributed or --data-parallel, not both.")

    rank = 0
    world_size = 1
    local_rank = 0

    if args.distributed:
        rank, world_size, local_rank = setup_distributed_training()

    torch.manual_seed(args.seed + rank)

    train_loader, val_loader = build_dataloaders(args, rank, world_size)

    model = SyntheticConstraintModel()
    loss_fn = ConstraintAwareLoss(alpha=0.5, huber_delta=1.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "use_amp": True,
        },
        "logging": {
            "log_frequency": 0,
            "epoch_log_frequency": 1,
        },
        "data": {
            "out_path": str(output_dir / "checkpoints"),
        },
    }

    trainer = DataParallelTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        rank=rank,
        world_size=world_size,
        device="auto",
        use_data_parallel=args.data_parallel,
    )

    history = trainer.train(train_loader, val_loader)

    if args.distributed and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        result = {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "best_val_loss": history["best_val_loss"],
            "total_epochs": history["total_epochs"],
            "model_hash": vector_hash(trainer.model.module if hasattr(trainer.model, "module") else trainer.model),
            "world_size": world_size,
            "data_parallel": args.data_parallel,
            "distributed": args.distributed,
            "gradient_accumulation": args.gradient_accumulation,
        }

        result_path = output_dir / args.result_json
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Synthetic run complete. Metrics written to {result_path}")
        print(json.dumps(result, indent=2))

    trainer.cleanup()


if __name__ == "__main__":
    main()

