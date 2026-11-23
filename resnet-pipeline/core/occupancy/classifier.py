"""ResNet-18 workstation-crop classifier.

Loads a fine-tuned torchvision ResNet-18 checkpoint safely (``weights_only=True``),
reproduces the training evaluation preprocessing exactly, and classifies batches of
BGR OpenCV crops.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

logger = logging.getLogger(__name__)

STATE_DICT_KEYS = ("model_state_dict", "state_dict")
RESERVED_CLASS_NAMES = ("unknown",)
INTERPOLATIONS = {
    "nearest": transforms.InterpolationMode.NEAREST,
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
}

FULL_MODEL_HELP = (
    "The checkpoint could not be loaded with torch.load(weights_only=True). It probably "
    "stores a pickled full model object, which is not loaded automatically because "
    "unpickling can execute arbitrary code. If you trust the file, convert it once:\n"
    "    import torch\n"
    "    model = torch.load('model.pth', map_location='cpu', weights_only=False)\n"
    "    torch.save({'state_dict': model.state_dict(), "
    "'class_to_idx': {'empty': 0, 'occupied': 1}}, 'converted.pth')\n"
    "(use your real class_to_idx from training)."
)


class CheckpointError(RuntimeError):
    """The checkpoint is unreadable, incompatible, or lacks required metadata."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Deterministic evaluation transform: Resize(shorter side) -> CenterCrop -> Normalize."""

    resize: int
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bilinear"
    color_order: str = "RGB"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PreprocessingConfig":
        required = ("resize", "crop_size", "mean", "std")
        missing = [k for k in required if data.get(k) is None]
        if missing:
            raise CheckpointError(f"preprocessing_config is missing {missing}")
        unknown = set(data) - {*required, "interpolation", "color_order"}
        if unknown:
            raise CheckpointError(f"preprocessing_config has unknown keys {sorted(unknown)}")
        config = cls(
            resize=int(data["resize"]),
            crop_size=int(data["crop_size"]),
            mean=tuple(float(v) for v in data["mean"]),  # type: ignore[arg-type]
            std=tuple(float(v) for v in data["std"]),  # type: ignore[arg-type]
            interpolation=str(data.get("interpolation", "bilinear")).lower(),
            color_order=str(data.get("color_order", "RGB")).upper(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.resize <= 0 or self.crop_size <= 0 or self.crop_size > self.resize:
            raise CheckpointError(f"Invalid resize/crop_size: {self.resize}/{self.crop_size}")
        if len(self.mean) != 3 or len(self.std) != 3 or any(s <= 0 for s in self.std):
            raise CheckpointError("mean and std must each have 3 values and std must be positive")
        if self.interpolation not in INTERPOLATIONS:
            raise CheckpointError(f"interpolation must be one of {sorted(INTERPOLATIONS)}")
        if self.color_order not in ("RGB", "BGR"):
            raise CheckpointError("color_order must be RGB or BGR")

    def build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.resize, interpolation=INTERPOLATIONS[self.interpolation]),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    logits: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


class CropClassifier(Protocol):
    """Interface the pipeline needs; implemented by ``ResNet18Classifier`` and test doubles."""

    class_names: list[str]

    def classify(self, crops: Mapping[str, np.ndarray]) -> dict[str, Prediction]: ...


def load_checkpoint(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise CheckpointError(f"{FULL_MODEL_HELP}\nOriginal error: {exc}") from exc
    except Exception as exc:
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc


def _is_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and bool(obj) and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    """Return ``(state_dict, location)`` for supported checkpoint layouts."""
    if _is_state_dict(checkpoint):
        state, location = dict(checkpoint), "<top level>"
    elif isinstance(checkpoint, Mapping):
        for key in STATE_DICT_KEYS:
            if _is_state_dict(checkpoint.get(key)):
                state, location = dict(checkpoint[key]), key
                break
        else:
            raise CheckpointError(
                f"No state dict found. Expected one of {STATE_DICT_KEYS} or a plain state dict; "
                f"top-level keys are {sorted(map(str, checkpoint))}"
            )
    else:
        raise CheckpointError(f"Unsupported checkpoint type {type(checkpoint).__name__}")
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
        location += " (DataParallel 'module.' prefix removed)"
    return state, location


def output_dimension(state: Mapping[str, torch.Tensor]) -> int:
    weight = state.get("fc.weight")
    if weight is None or weight.ndim != 2:
        raise CheckpointError("State dict has no 2-D 'fc.weight'; this is not a torchvision ResNet-18 head")
    if weight.shape[1] != 512:
        raise CheckpointError(f"fc.weight has input dim {weight.shape[1]}, ResNet-18 expects 512")
    return int(weight.shape[0])


def class_names_from_metadata(checkpoint: Any) -> list[str] | None:
    """Ordered class names (index -> name) from checkpoint metadata, or None if absent."""
    if not isinstance(checkpoint, Mapping):
        return None
    if isinstance(checkpoint.get("class_to_idx"), Mapping):
        mapping = {str(name): int(index) for name, index in checkpoint["class_to_idx"].items()}
        return _ordered(mapping, "class_to_idx")
    if isinstance(checkpoint.get("idx_to_class"), Mapping):
        mapping = {str(name): int(index) for index, name in checkpoint["idx_to_class"].items()}
        return _ordered(mapping, "idx_to_class")
    for key in ("class_names", "classes"):
        if isinstance(checkpoint.get(key), (list, tuple)):
            return [str(name) for name in checkpoint[key]]
    return None


def _ordered(name_to_index: Mapping[str, int], source: str) -> list[str]:
    indices = sorted(name_to_index.values())
    if indices != list(range(len(indices))):
        raise CheckpointError(f"{source} indices must be 0..N-1 without gaps, got {dict(name_to_index)}")
    return [name for name, _ in sorted(name_to_index.items(), key=lambda item: item[1])]


def resolve_class_names(checkpoint: Any, override: Sequence[str] | None, num_outputs: int) -> list[str]:
    metadata = class_names_from_metadata(checkpoint)
    if override is not None:
        names = [str(n) for n in override]
        if metadata is not None and metadata != names:
            raise CheckpointError(
                f"model.class_names {names} disagrees with checkpoint metadata {metadata} (index order). "
                "Remove the override or fix it."
            )
    elif metadata is not None:
        names = metadata
    else:
        raise CheckpointError(
            "Checkpoint has no class mapping (class_to_idx / idx_to_class / class_names). "
            "Set model.class_names in settings.yaml in exact training index order, e.g. "
            "[empty, occupied] for torchvision ImageFolder (alphabetical folder names)."
        )
    if len(names) != num_outputs:
        raise CheckpointError(f"{len(names)} class names {names} but the model has {num_outputs} outputs")
    if len(set(names)) != len(names) or any(not n.strip() for n in names):
        raise CheckpointError(f"Class names must be unique and non-empty: {names}")
    reserved = [n for n in names if n.lower() in RESERVED_CLASS_NAMES]
    if reserved:
        raise CheckpointError(f"Class names {reserved} collide with the internal UNKNOWN monitoring state")
    return names


def resolve_preprocessing(checkpoint: Any, override: Mapping[str, Any] | None) -> PreprocessingConfig:
    """Use explicit preprocessing (config or checkpoint 'preprocessing' dict) and cross-check it."""
    embedded = checkpoint.get("preprocessing") if isinstance(checkpoint, Mapping) else None
    source = override if override is not None else embedded
    if source is None:
        found = {k: checkpoint[k] for k in ("image_size", "mean", "std")
                 if isinstance(checkpoint, Mapping) and k in checkpoint}
        raise CheckpointError(
            "Training preprocessing is not fully described by the checkpoint "
            f"(found {found or 'nothing'}). Set model.preprocessing_config in settings.yaml with "
            "resize (shorter side), crop_size, mean, std, interpolation, matching your training eval transform."
        )
    config = PreprocessingConfig.from_mapping(source)
    if isinstance(checkpoint, Mapping):
        size = checkpoint.get("image_size")
        if size is not None and int(size) != config.crop_size:
            raise CheckpointError(f"crop_size {config.crop_size} != checkpoint image_size {size}")
        for key in ("mean", "std"):
            value = checkpoint.get(key)
            if value is not None and not np.allclose(value, getattr(config, key), atol=1e-6):
                raise CheckpointError(f"{key} {getattr(config, key)} != checkpoint {key} {tuple(value)}")
    return config


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise CheckpointError(f"device '{device}' requested but CUDA is not available")
    return torch.device(device)


def build_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18Classifier:
    """Batch classifier for BGR workstation crops."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        class_names: Sequence[str] | None = None,
        preprocessing: Mapping[str, Any] | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        checkpoint = load_checkpoint(checkpoint_path)
        name = checkpoint.get("model_name") if isinstance(checkpoint, Mapping) else None
        if name is not None and str(name).lower() != "resnet18":
            raise CheckpointError(f"Checkpoint model_name is '{name}', expected 'resnet18'")
        state, location = extract_state_dict(checkpoint)
        num_outputs = output_dimension(state)
        self.class_names = resolve_class_names(checkpoint, class_names, num_outputs)
        self.preprocessing = resolve_preprocessing(checkpoint, preprocessing)
        self._transform = self.preprocessing.build_transform()

        model = build_resnet18(num_outputs)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise CheckpointError(f"Weights are not compatible with ResNet-18/{num_outputs} classes: {exc}") from exc
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.metadata = {
            k: v for k, v in (checkpoint.items() if isinstance(checkpoint, Mapping) else [])
            if isinstance(v, (int, float, str)) and k not in STATE_DICT_KEYS
        }
        logger.info(
            "Loaded ResNet-18 from %s (state dict: %s) classes=%s device=%s preprocessing=%s",
            checkpoint_path, location, self.class_names, self.device, self.preprocessing,
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """BGR uint8 crop -> normalized CHW tensor, identical to the training eval transform."""
        if crop is None or crop.size == 0 or crop.ndim not in (2, 3):
            raise ValueError("Empty or malformed crop")
        if crop.dtype != np.uint8:
            raise ValueError(f"Crops must be uint8, got {crop.dtype}")
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        elif crop.shape[2] == 3:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count {crop.shape[2]}")
        if self.preprocessing.color_order == "BGR":
            rgb = rgb[:, :, ::-1].copy()
        return self._transform(Image.fromarray(rgb))

    def preprocess_batch(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.stack([self.preprocess(crop) for crop in crops])

    def predict_tensor(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run preprocessed batches through the model; returns CPU ``(logits, probabilities)``."""
        logits_out, probs_out = [], []
        with torch.inference_mode():
            for start in range(0, batch.shape[0], self.batch_size):
                chunk = batch[start:start + self.batch_size]
                if self.device.type == "cuda":
                    chunk = chunk.pin_memory().to(self.device, non_blocking=True)
                logits = self.model(chunk)
                probs = torch.softmax(logits.float(), dim=1)
                logits_out.append(logits.float().cpu())
                probs_out.append(probs.cpu())
        return torch.cat(logits_out), torch.cat(probs_out)

    def predict(self, crops: Sequence[np.ndarray]) -> list[Prediction]:
        if not crops:
            return []
        logits, probs = self.predict_tensor(self.preprocess_batch(crops))
        results = []
        for row_logits, row_probs in zip(logits.tolist(), probs.tolist()):
            index = int(np.argmax(row_probs))
            if not all(math.isfinite(p) for p in row_probs):
                raise RuntimeError("Model produced non-finite probabilities")
            results.append(Prediction(
                predicted_class=self.class_names[index],
                confidence=float(row_probs[index]),
                probabilities={name: float(p) for name, p in zip(self.class_names, row_probs)},
                logits=[float(v) for v in row_logits],
            ))
        return results

    def classify(self, crops: Mapping[str, np.ndarray]) -> dict[str, Prediction]:
        ids = list(crops)
        return dict(zip(ids, self.predict([crops[i] for i in ids])))
