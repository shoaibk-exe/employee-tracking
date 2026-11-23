"""Inspect a ResNet-18 checkpoint safely and report what inference still needs.

    python -m scripts.occupancy.inspect_checkpoint --checkpoint models/resnet18_empty_occupied.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from core.occupancy.classifier import (
    CheckpointError, build_resnet18, class_names_from_metadata, extract_state_dict, load_checkpoint,
    output_dimension, resolve_class_names, resolve_preprocessing,
)


def describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, dict):
        return f"dict ({len(value)} entries)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}{list(value) if len(value) <= 8 else f' (len {len(value)})'}"
    return f"{type(value).__name__} = {value!r}"


def inspect(path: Path, settings_path: Path | None) -> int:
    print(f"Checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)" if path.is_file() else f"Missing: {path}")
    try:
        checkpoint = load_checkpoint(path)
    except CheckpointError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("Loaded with torch.load(weights_only=True): OK")

    print("\n== Structure")
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            print(f"  {key}: {describe(value) if key not in ('state_dict', 'model_state_dict') else describe(value)}")
    else:
        print(f"  {type(checkpoint).__name__}")

    missing: list[str] = []
    print("\n== Weights")
    try:
        state, location = extract_state_dict(checkpoint)
        print(f"  State dict: {location}, {len(state)} tensors")
        num_outputs = output_dimension(state)
        print(f"  Classifier head fc.weight: {tuple(state['fc.weight'].shape)} -> {num_outputs} output classes")
        model = build_resnet18(num_outputs)
        model.load_state_dict(state, strict=True)
        print("  Strict load into torchvision resnet18: OK")
    except (CheckpointError, RuntimeError) as exc:
        print(f"  ERROR: {exc}")
        return 1

    print("\n== Class metadata")
    names = None
    try:
        names = class_names_from_metadata(checkpoint)
    except CheckpointError as exc:
        print(f"  ERROR: {exc}")
    if names:
        print("  Index -> class: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
        if len(names) != num_outputs:
            print(f"  ERROR: {len(names)} names but {num_outputs} outputs")
    else:
        print("  None found")
        missing.append("model.class_names: list of class names in training index order")

    print("\n== Preprocessing metadata")
    found = {k: checkpoint[k] for k in ("image_size", "mean", "std", "preprocessing")
             if isinstance(checkpoint, dict) and k in checkpoint}
    for key, value in found.items():
        print(f"  {key}: {value}")
    if not found:
        print("  None found")
    if not (isinstance(checkpoint, dict) and isinstance(checkpoint.get("preprocessing"), dict)):
        missing.append("model.preprocessing_config: resize (shorter side), crop_size, interpolation, "
                       "color_order, mean, std of the training EVAL transform (resize is not stored)")

    other = {k: v for k, v in checkpoint.items()
             if isinstance(checkpoint, dict) and k not in ("state_dict", "model_state_dict", "class_to_idx",
                                                            "idx_to_class", "class_names", "classes",
                                                            "image_size", "mean", "std", "preprocessing")}
    if other:
        print("\n== Other metadata (reported by training, not verified here)")
        for key, value in other.items():
            print(f"  {key}: {value}")

    if settings_path is not None and settings_path.is_file():
        print(f"\n== Check against {settings_path}")
        model_cfg = (yaml.safe_load(settings_path.read_text()) or {}).get("model", {})
        try:
            resolved = resolve_class_names(checkpoint, model_cfg.get("class_names"), num_outputs)
            print(f"  Class names: {resolved}")
            pre = resolve_preprocessing(checkpoint, model_cfg.get("preprocessing_config"))
            print(f"  Preprocessing: {pre}")
            print("  Ready for inference.")
            missing = []
        except CheckpointError as exc:
            print(f"  ERROR: {exc}")

    if missing:
        print("\n== You still need to supply (in config/settings.yaml):")
        for item in missing:
            print(f"  - {item}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a ResNet-18 checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_empty_occupied.pth"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"),
                        help="Settings file used to fill metadata gaps (use --config '' to skip)")
    args = parser.parse_args(argv)
    return inspect(args.checkpoint, args.config if str(args.config) else None)


if __name__ == "__main__":
    sys.exit(main())
