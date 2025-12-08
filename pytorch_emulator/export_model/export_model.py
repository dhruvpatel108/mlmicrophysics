import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.physics_emulator import ConstraintAwareEmulator

OUTPUT_TENSOR_ORDER: Sequence[str] = (
    "qrtend",
    "nctend",
    "nrtend",
    "qctend",
)


class E3SMWrapper(nn.Module):
    """Thin wrapper that flattens the emulator dict into a single tensor."""

    def __init__(self, original_model: nn.Module) -> None:
        super().__init__()
        self.model = original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        stacked = [outputs[name] for name in OUTPUT_TENSOR_ORDER]
        return torch.cat(stacked, dim=1)


def load_model_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    import yaml
    with config_path.open("r") as fp:
        config = yaml.safe_load(fp)

    if not isinstance(config, dict):
        raise ValueError(f"Unexpected config format in {config_path}")

    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Expected `model` section in config to be a mapping; "
            f"found {type(model_cfg).__name__}"
        )
    return model_cfg


def instantiate_model(model_cfg: Dict[str, Any]) -> ConstraintAwareEmulator:
    ctor_kwargs: Dict[str, Any] = {}
    for key in ("input_dim", "shared_dims", "head_dims", "dropout"):
        if key in model_cfg:
            ctor_kwargs[key] = model_cfg[key]

    dropout = float(ctor_kwargs.get("dropout", 0.0))
    if dropout != 0.0:
        print(
            f"[info] Dropout={dropout} detected in config; "
            "forcing to 0.0 for export."
        )
        ctor_kwargs["dropout"] = 0.0

    return ConstraintAwareEmulator(**ctor_kwargs)


def resolve_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def verify_equivalence(
    eager_module: nn.Module,
    scripted_module: torch.jit.ScriptModule,
    batch_size: int,
    input_dim: int,
    device: torch.device,
) -> float:
    sample = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        eager_out = eager_module(sample).cpu()
        scripted_out = scripted_module(sample).cpu()

    max_diff = torch.max(torch.abs(eager_out - scripted_out)).item()
    print(f"[verify] max |Δ| between eager and TorchScript outputs: {max_diff:.3e}")
    return max_diff


def parse_args() -> argparse.Namespace:
    sample_run_dir = THIS_DIR / "run_107666"
    default_checkpoint: Optional[Path] = sample_run_dir / "latest_checkpoint.pth"
    if not default_checkpoint.exists():
        default_checkpoint = None
    default_config: Optional[Path] = sample_run_dir / "config_used.yml"
    if not default_config.exists():
        default_config = None

    parser = argparse.ArgumentParser(
        description="Trace the ConstraintAwareEmulator into a TorchScript artifact "
        "compatible with FTorch/E3SM."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=default_checkpoint is None,
        default=default_checkpoint,
        help=(
            "Path to the training checkpoint. "
            "Uses run_107666/latest_checkpoint.pth when present."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=(
            "Training config with model hyperparameters. "
            "Uses run_107666/config_used.yml when present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=THIS_DIR / "emulator_for_e3sm.pt",
        help="Destination TorchScript file.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to instantiate the model on during tracing.",
    )
    parser.add_argument(
        "--trace-batch",
        type=int,
        default=16,
        help="Batch size to use for tracing and verification inputs.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip comparing eager vs TorchScript outputs after export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config and not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    model_cfg = load_model_config(args.config)
    model = instantiate_model(model_cfg)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = resolve_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    wrapper = E3SMWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(args.trace_batch, model_cfg.get("input_dim", model.input_dim), device=device)
    traced = torch.jit.trace(wrapper, dummy_input, strict=False)
    frozen_model = torch.jit.freeze(traced)
    frozen_model.save(str(args.output))

    print(f"[done] TorchScript module saved to {args.output}")
    print("[info] Output tensor columns:")
    for idx, name in enumerate(OUTPUT_TENSOR_ORDER):
        print(f"  {idx}: {name}")

    if not args.skip_verify:
        scripted = torch.jit.load(str(args.output), map_location=device)
        max_diff = verify_equivalence(
            wrapper, scripted, args.trace_batch, dummy_input.shape[1], device
        )
        if max_diff > 1e-4:
            print(
                "[warn] Difference between eager and scripted outputs is larger than "
                "1e-4. Inspect before handing off to FTorch."
            )


if __name__ == "__main__":
    main()