"""Classify workstation crop images with the configured checkpoint.

    python -m scripts.occupancy.test_classifier --image path/to/workstation_crop.jpg
    python -m scripts.occupancy.test_classifier --dir training/resnet-18/dataset/test   # class-folder evaluation

Images are read with OpenCV (BGR), exactly like video crops. ``--compare-training``
also runs the training script's PIL eval pipeline on the same files and reports the
largest probability difference, which verifies the inference preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from core.occupancy.classifier import CheckpointError, ResNet18Classifier
from core.config import DEFAULTS, ConfigError, _merge

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classifier(args: argparse.Namespace) -> ResNet18Classifier:
    cfg = DEFAULTS
    if args.config and Path(args.config).is_file():
        cfg = _merge(DEFAULTS, yaml.safe_load(Path(args.config).read_text()) or {})
    model = cfg["model"]
    return ResNet18Classifier(
        args.checkpoint or model["checkpoint"], model["class_names"], model["preprocessing_config"],
        args.device or model["device"], model["batch_size"],
    )


def training_pipeline_probs(classifier: ResNet18Classifier, paths: list[Path]) -> np.ndarray:
    """Training eval transform on PIL-decoded files (exif_transpose + RGB), as in train.py."""
    tensors = [classifier._transform(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths]
    _, probs = classifier.predict_tensor(torch.stack(tensors))
    return probs.numpy()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify workstation crops with ResNet-18")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, nargs="+")
    group.add_argument("--dir", type=Path, help="Folder with one subfolder per class (ImageFolder layout)")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--compare-training", action="store_true",
                        help="Also run the train.py PIL pipeline and report the max probability difference")
    args = parser.parse_args(argv)

    try:
        classifier = load_classifier(args)
    except (CheckpointError, ConfigError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"classes (index order): {classifier.class_names}  device: {classifier.device}")

    if args.image:
        paths, labels = list(args.image), None
    else:
        folders = sorted(p for p in args.dir.iterdir() if p.is_dir())
        unknown = [f.name for f in folders if f.name not in classifier.class_names]
        if unknown:
            print(f"ERROR: folders {unknown} are not model classes {classifier.class_names}")
            return 2
        paths, labels = [], []
        for folder in folders:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            paths += files
            labels += [folder.name] * len(files)

    predictions = classifier.predict([read_bgr(p) for p in paths])
    if args.image:
        for path, pred in zip(paths, predictions):
            probs = "  ".join(f"{k}: {v:.3f}" for k, v in pred.probabilities.items())
            print(f"{path}: {pred.predicted_class} {pred.confidence:.3f}   [{probs}]")
    else:
        names = classifier.class_names
        matrix = np.zeros((len(names), len(names)), dtype=int)
        for label, pred in zip(labels, predictions):
            matrix[names.index(label), names.index(pred.predicted_class)] += 1
        correct = int(np.trace(matrix))
        print(f"\nMeasured on {len(paths)} images: accuracy {correct / len(paths):.4f} ({correct}/{len(paths)})")
        print("confusion (rows = true, cols = predicted): " + ", ".join(names))
        for name, row in zip(names, matrix):
            print(f"  {name:>14}: {row.tolist()}")
        wrong = [(p, l, pr) for p, l, pr in zip(paths, labels, predictions) if pr.predicted_class != l]
        for path, label, pred in wrong[:20]:
            print(f"  MISS {path.name}: true {label}, predicted {pred.predicted_class} {pred.confidence:.3f}")

    if args.compare_training:
        reference = training_pipeline_probs(classifier, paths)
        ours = np.array([[p.probabilities[n] for n in classifier.class_names] for p in predictions])
        diff = float(np.abs(reference - ours).max())
        agree = int((reference.argmax(1) == ours.argmax(1)).sum())
        print(f"\nTraining-pipeline comparison: max |prob diff| = {diff:.6f}, "
              f"same predicted class on {agree}/{len(paths)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
