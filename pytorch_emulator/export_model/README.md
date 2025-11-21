# TorchScript Export Guide

This directory bundles everything needed to turn a trained microphysics emulator into a TorchScript artifact that can be embedded in E3SM via FTorch.

## At a Glance
- **Script:** `export_model.py`
- **Outputs:** `emulator_for_e3sm.pt` (TorchScript) + console summary of tensor column order
- **Dependencies:** PyTorch ≥ 1.13, PyYAML (for reading the training config)
- **Sample assets:** `run_107666/config_used.yml`, `run_107666/latest_checkpoint.pth`

## Quick Start
```bash
# Default demo run (CPU)
python export_model.py

# Custom checkpoint/config
python export_model.py \
  --checkpoint /path/to/checkpoint.pth \
  --config /path/to/config.yml \
  --output /desired/path/emulator_for_e3sm.pt
```

Helpful flags:
- `--device {cpu,cuda}` — trace on GPU if available
- `--trace-batch N` — batch size for tracing/verification dummy inputs (default 16)
- `--skip-verify` — skip post-export eager vs TorchScript consistency check (not recommended unless debugging)

## What the Script Does
1. Loads the training configuration to reconstruct the model architecture (`ConstraintAwareEmulator`).
2. Forces dropout to zero for deterministic inference and loads checkpoint weights.
3. Wraps the model in `E3SMWrapper`, flattening outputs into a single tensor with documented ordering:
   0. `qrtend`
   1. `nctend`
   2. `nrtend`
   3. `qctend`
   4. `is_active`
4. Traces + freezes the model with `torch.jit.trace` and writes the TorchScript module to disk.
5. Reloads the artifact and compares it with the eager model unless `--skip-verify` is supplied.

## Sharing with FTorch / E3SM Teams
- Provide `emulator_for_e3sm.pt` together with a note on the column ordering above.
- Include the exact config file and checkpoint (or commit hash) used during export for reproducibility.
- If your collaborators use FTorch, point them to the documentation at https://github.com/Cambridge-ICCS/FTorch for loading TorchScript modules in Fortran/C++.

## Sanity Checks Before Hand-off
- Run the script once without `--skip-verify` and confirm the max |Δ| is near machine precision.
- Optionally, open a Python session and load the saved TorchScript module to inspect example outputs.
- Keep an eye on the console output for the reported tensor column order.

Questions or issues? Drop a note in the repo along with the command you ran and the checkpoint/config you used.
