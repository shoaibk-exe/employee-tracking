#!/usr/bin/env python3
"""Fine-tune VideoMAE-Large on the trimmed activity clips.

Classes are the subfolder names under trims/ (computer_interaction, neither,
phone_using). Source videos are only read.

The checkpoint is MCG-NJU/videomae-large-finetuned-kinetics: the Kinetics-400
VideoMAE-Large model (24 layers, 1024-d). That is the strongest VideoMAE that
fits a 12GB GPU on this small set. VideoMAE-Huge does not train reliably in
12GB and memorizes a few dozen clips.

Train from the videomae folder:

    /home/ai-dw/miniconda3/envs/y3/bin/python train.py

The best model is written to checkpoints/videomae-large-activity/.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent
TRIMS = ROOT / "trims"
CHECKPOINT_DIR = ROOT / "checkpoints" / "videomae-large-activity"
MODEL_ID = "MCG-NJU/videomae-large-finetuned-kinetics"

NUM_FRAMES = 16
FRAME_STRIDE = 4
CLIPS_PER_VIDEO = 8
VAL_CLIPS = 5
HEAD_EPOCHS = 5
FULL_EPOCHS = 25
PATIENCE = 8
BATCH_SIZE = 2
GRAD_ACCUM = 2
HEAD_LR = 1e-3
FULL_LR = 2e-4
LAYER_DECAY = 0.75
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_RATIO = 0.2
SEED = 42
NUM_WORKERS = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_classes() -> list[str]:
    classes = sorted(p.name for p in TRIMS.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders in {TRIMS}")
    return classes


def list_videos(classes: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for name in classes:
        files = sorted((TRIMS / name).glob("*.mp4"))
        if not files:
            raise SystemExit(f"No mp4 files in {TRIMS / name}")
        grouped[name] = files
    return grouped


def stratified_split(grouped: dict[str, list[Path]], seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Hold out ~20% of videos per class. A video never appears in both splits."""
    rng = random.Random(seed)
    train, val = [], []
    for label, name in enumerate(sorted(grouped)):
        files = grouped[name][:]
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * VAL_RATIO)))
        n_val = min(n_val, len(files) - 1)
        for path in files[:n_val]:
            val.append((path, label))
        for path in files[n_val:]:
            train.append((path, label))
    return train, val


def load_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return np.stack(frames, axis=0)


def clip_span() -> int:
    return (NUM_FRAMES - 1) * FRAME_STRIDE + 1


def max_start(num_frames: int) -> int:
    return max(0, num_frames - clip_span())


def frame_indices(num_frames: int, start: int) -> list[int]:
    if num_frames <= 1:
        return [0] * NUM_FRAMES
    if num_frames < clip_span():
        return np.linspace(0, num_frames - 1, NUM_FRAMES).astype(int).tolist()
    return [min(num_frames - 1, start + i * FRAME_STRIDE) for i in range(NUM_FRAMES)]


def val_starts(num_frames: int) -> list[int]:
    hi = max_start(num_frames)
    if hi == 0:
        return [0]
    if VAL_CLIPS == 1:
        return [hi // 2]
    return sorted({int(round(i * hi / (VAL_CLIPS - 1))) for i in range(VAL_CLIPS)})


def augment_clip(frames: list[np.ndarray], train: bool, rng: random.Random) -> list[np.ndarray]:
    if not train:
        return frames
    if rng.random() < 0.5:
        frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.8, 1.2)
    adjusted = []
    for frame in frames:
        image = frame.astype(np.float32)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean
        image *= brightness
        adjusted.append(np.clip(image, 0, 255).astype(np.uint8))
    scale = rng.uniform(0.85, 1.0)
    height, width = adjusted[0].shape[:2]
    crop_h = max(16, int(round(height * scale)))
    crop_w = max(16, int(round(width * scale)))
    top = rng.randint(0, height - crop_h)
    left = rng.randint(0, width - crop_w)
    return [frame[top : top + crop_h, left : left + crop_w] for frame in adjusted]


class ClipDataset(Dataset):
    def __init__(
        self,
        clips: list[tuple[str, int, int]],
        videos: dict[str, np.ndarray],
        processor: VideoMAEImageProcessor,
        train: bool,
    ) -> None:
        self.clips = clips
        self.videos = videos
        self.processor = processor
        self.train = train

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, start = self.clips[index]
        video = self.videos[path]
        indices = frame_indices(video.shape[0], start)
        frames = augment_clip([video[i] for i in indices], self.train, random.Random(SEED + index + start))
        packed = self.processor(frames, return_tensors="pt")
        pixels = packed["pixel_values"]
        if pixels.ndim == 5:
            pixels = pixels[0]
        return pixels, label


def build_train_clips(samples: list[tuple[Path, int]], lengths: dict[str, int]) -> list[tuple[str, int, int]]:
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path, label in samples:
        key = str(path)
        by_label[label].append((key, lengths[key]))
    per_class = max(len(group) for group in by_label.values()) * CLIPS_PER_VIDEO
    clips: list[tuple[str, int, int]] = []
    for label, group in by_label.items():
        for _ in range(per_class):
            key, nframes = random.choice(group)
            clips.append((key, label, random.randint(0, max_start(nframes))))
    random.shuffle(clips)
    return clips


def build_val_clips(samples: list[tuple[Path, int]], lengths: dict[str, int]) -> list[tuple[str, int, int]]:
    clips = []
    for path, label in samples:
        key = str(path)
        for start in val_starts(lengths[key]):
            clips.append((key, label, start))
    return clips


def no_weight_decay(name: str) -> bool:
    return name.endswith(".bias") or "norm" in name.lower() or "pos_embed" in name


def optimizer_groups(model: nn.Module, base_lr: float, head_only: bool) -> list[dict]:
    num_layers = model.config.num_hidden_layers
    buckets: dict[tuple[int, bool], list[nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier") or name.startswith("fc_norm"):
            layer_id = num_layers + 1
        elif "encoder.layer." in name:
            layer_id = int(name.split("encoder.layer.")[1].split(".")[0]) + 1
        else:
            layer_id = 0
        if head_only:
            scale = 1.0
        else:
            scale = LAYER_DECAY ** (num_layers + 1 - layer_id)
        buckets[(layer_id, no_weight_decay(name))].append((param, scale))

    groups = []
    for (layer_id, skip_decay), params in sorted(buckets.items()):
        scale = params[0][1]
        groups.append(
            {
                "params": [param for param, _ in params],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if skip_decay else WEIGHT_DECAY,
            }
        )
    return groups


def restore_attention_bias(model: nn.Module) -> None:
    """Copy Kinetics q_bias/v_bias into the Linear biases this Transformers build expects.

    The checkpoint stores query and value bias as q_bias and v_bias. There is no key
    bias, so key.bias stays zero. Without this, attention starts from a random bias.
    """
    raw = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))
    mapped = {}
    for key, value in raw.items():
        if key.endswith(".attention.attention.q_bias"):
            mapped[key.replace(".q_bias", ".query.bias")] = value
        elif key.endswith(".attention.attention.v_bias"):
            mapped[key.replace(".v_bias", ".value.bias")] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected bias keys: {unexpected[:4]}")
    restored = [key for key in missing if key.endswith(".query.bias") or key.endswith(".value.bias")]
    if restored:
        raise RuntimeError(f"Attention bias was not restored: {restored[:4]}")
    zeros = 0
    for name, module in model.named_modules():
        if name.endswith("attention.attention.key") and getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
            zeros += 1
    if zeros != model.config.num_hidden_layers:
        raise RuntimeError(f"Expected {model.config.num_hidden_layers} key biases, zeroed {zeros}")
    print(f"Restored pretrained q/v attention bias on {zeros} layers; key bias is 0")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("classifier") or name.startswith("fc_norm"):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def class_weights(samples: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    counts = np.bincount([label for _, label in samples], minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def macro_f1(truth: list[int], pred: list[int], num_classes: int) -> float:
    scores = []
    for label in range(num_classes):
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


def per_class_recall(truth: list[int], pred: list[int], classes: list[str]) -> dict[str, float]:
    recalls = {}
    for label, name in enumerate(classes):
        total = sum(t == label for t in truth)
        hit = sum(t == label and p == label for t, p in zip(truth, pred))
        recalls[name] = hit / total if total else 0.0
    return recalls


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return pixels, labels


def apply_epoch_lr(optimizer: torch.optim.Optimizer, epoch: int, epochs: int) -> None:
    """One warmup epoch, then cosine decay. Each param group keeps its layer-wise base lr."""
    if epoch <= 1:
        scale = 1.0
    else:
        progress = (epoch - 1) / max(1, epochs - 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    weights: torch.Tensor,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    seen = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for step, (pixels, labels) in enumerate(loader, start=1):
        pixels = pixels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixels).logits
                loss = nn.functional.cross_entropy(logits, labels, weight=weights, label_smoothing=LABEL_SMOOTHING)
            if train:
                (loss / GRAD_ACCUM).backward()
                if step % GRAD_ACCUM == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    samples: list[tuple[Path, int]],
    videos: dict[str, np.ndarray],
    processor: VideoMAEImageProcessor,
    classes: list[str],
    device: torch.device,
) -> dict:
    model.eval()
    truth, pred = [], []
    for path, label in samples:
        key = str(path)
        video = videos[key]
        logits_sum = None
        for start in val_starts(video.shape[0]):
            indices = frame_indices(video.shape[0], start)
            frames = [video[i] for i in indices]
            pixels = processor(frames, return_tensors="pt")["pixel_values"]
            if pixels.ndim == 4:
                pixels = pixels.unsqueeze(0)
            pixels = pixels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixels).logits
            logits_sum = logits.float() if logits_sum is None else logits_sum + logits.float()
        guess = int(logits_sum.argmax(dim=-1).item())
        truth.append(label)
        pred.append(guess)
    correct = sum(t == p for t, p in zip(truth, pred))
    return {
        "accuracy": correct / len(truth),
        "macro_f1": macro_f1(truth, pred, len(classes)),
        "recall": per_class_recall(truth, pred, classes),
        "n": len(truth),
    }


def save_checkpoint(model, processor, classes: list[str], metrics: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    (CHECKPOINT_DIR / "metrics.json").write_text(json.dumps({"classes": classes, **metrics}, indent=2) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to train VideoMAE-Large.")
    set_seed(SEED)
    device = torch.device("cuda")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    classes = discover_classes()
    grouped = list_videos(classes)
    train_samples, val_samples = stratified_split(grouped, SEED)
    history = []
    best_f1 = -1.0
    stale = 0
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_labels = len(classes)
    config.id2label = {str(i): name for i, name in enumerate(classes)}
    config.label2id = {name: i for i, name in enumerate(classes)}
    config.problem_type = "single_label_classification"
    print(f"Loading {MODEL_ID}")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )
    restore_attention_bias(model)
    split_record = {
        "model": MODEL_ID,
        "seed": SEED,
        "classes": classes,
        "train": [str(path.relative_to(TRIMS)) for path, _ in train_samples],
        "val": [str(path.relative_to(TRIMS)) for path, _ in val_samples],
    }
    (CHECKPOINT_DIR / "split.json").write_text(json.dumps(split_record, indent=2) + "\n")

    print(f"Classes: {classes}")
    print(f"Train videos: {len(train_samples)}  Val videos: {len(val_samples)}")
    for name in classes:
        n_train = sum(path.parent.name == name for path, _ in train_samples)
        n_val = sum(path.parent.name == name for path, _ in val_samples)
        print(f"  {name}: train {n_train}  val {n_val}")

    print("Decoding videos into memory...")
    videos = {str(path): load_video(path) for path, _ in train_samples + val_samples}
    lengths = {key: int(value.shape[0]) for key, value in videos.items()}
    print(f"  {len(videos)} videos, frames {min(lengths.values())}-{max(lengths.values())}")

    model.gradient_checkpointing_enable()
    model.to(device)
    weights = class_weights(train_samples, len(classes)).to(device)

    def train_stage(title: str, epochs: int, base_lr: float, head_only: bool) -> None:
        nonlocal best_f1, stale, history
        set_backbone_trainable(model, trainable=not head_only)
        optimizer = torch.optim.AdamW(optimizer_groups(model, base_lr, head_only=head_only))
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        print(f"\n{title}: epochs 1-{epochs}, base lr {base_lr:g}")
        for epoch in range(1, epochs + 1):
            set_seed(SEED + epoch + (0 if head_only else 1000))
            apply_epoch_lr(optimizer, epoch, epochs)
            clips = build_train_clips(train_samples, lengths)
            loader = DataLoader(
                ClipDataset(clips, videos, processor, train=True),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate,
            )
            loss = run_epoch(model, loader, optimizer, weights, device, train=True)
            metrics = evaluate(model, val_samples, videos, processor, classes, device)
            history.append({"stage": title, "epoch": epoch, "train_loss": loss, **metrics})
            recall = "  ".join(f"{name} {score:.2f}" for name, score in metrics["recall"].items())
            print(
                f"  epoch {epoch:02d}  loss {loss:.4f}  "
                f"val acc {metrics['accuracy']:.3f}  macro F1 {metrics['macro_f1']:.3f}  recall {recall}"
            )
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                stale = 0
                save_checkpoint(model, processor, classes, {"best_macro_f1": best_f1, "history": history})
                print(f"  saved {CHECKPOINT_DIR}")
            else:
                stale += 1
            if not head_only and stale >= PATIENCE:
                print(f"  early stop, no macro-F1 gain for {PATIENCE} epochs")
                return

    train_stage("head", HEAD_EPOCHS, HEAD_LR, head_only=True)
    best_model = VideoMAEForVideoClassification.from_pretrained(CHECKPOINT_DIR)
    model.load_state_dict(best_model.state_dict())
    del best_model
    stale = 0

    train_stage("full", FULL_EPOCHS, FULL_LR, head_only=False)
    print(f"\nBest val macro F1: {best_f1:.3f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
