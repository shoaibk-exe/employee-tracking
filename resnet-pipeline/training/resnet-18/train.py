#!/usr/bin/env python3
"""Fine-tune ImageNet ResNet-18 on the empty/occupied dataset.

Dataset layout (torchvision ImageFolder):
  dataset/train/{empty,occupied}
  dataset/val/{empty,occupied}
  dataset/test/{empty,occupied}

Writes the best validation checkpoint to
checkpoints/resnet18_empty_occupied.pth

Train:
  python train.py

Classify one image with the saved checkpoint:
  python train.py --image path/to/image.jpg
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
CHECKPOINT = ROOT / "checkpoints" / "resnet18_empty_occupied.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_transforms():
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img).convert("RGB")),
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img).convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def make_loader(split, transform, batch_size, shuffle):
    dataset = datasets.ImageFolder(DATA_DIR / split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, loader


def run_epoch(model, loader, criterion, optimizer, device, scaler, train):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu()
        for true, pred in zip(labels, preds):
            matrix[true, pred] += 1
    return matrix


def class_report(matrix, class_names):
    lines = []
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        hits = int(matrix[index, index])
        recall = hits / support if support else 0.0
        precision = hits / predicted if predicted else 0.0
        lines.append(
            f"  {name}: precision {precision:.3f}  recall {recall:.3f}  support {support}"
        )
    correct = int(matrix.diag().sum())
    total = int(matrix.sum())
    accuracy = correct / total if total else 0.0
    return accuracy, "\n".join(lines)


def save_checkpoint(path, model, class_to_idx, epoch, val_acc, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "resnet18",
            "state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": IMAGE_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "epoch": epoch,
            "val_acc": val_acc,
            "val_loss": val_loss,
        },
        path,
    )


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = make_transforms()
    train_ds, train_loader = make_loader("train", train_tf, args.batch_size, shuffle=True)
    val_ds, val_loader = make_loader("val", eval_tf, args.batch_size, shuffle=False)
    test_ds, test_loader = make_loader("test", eval_tf, args.batch_size, shuffle=False)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("Class folders do not match across train, val, and test.")

    class_to_idx = train_ds.class_to_idx
    class_names = train_ds.classes
    print(f"device {device}")
    print(f"classes {class_to_idx}")
    print(f"images train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    backbone = [param for name, param in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr},
            {"params": model.fc.parameters(), "lr": args.lr * 10},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc = -1.0
    best_loss = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False
        )
        scheduler.step()
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        mark = ""
        if improved:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(CHECKPOINT, model, class_to_idx, epoch, val_acc, val_loss)
            mark = "  saved"
        else:
            stale += 1
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}{mark}"
        )
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    matrix = confusion_matrix(model, test_loader, device, len(class_names))
    test_acc, report = class_report(matrix, class_names)
    checkpoint["test_acc"] = test_acc
    torch.save(checkpoint, CHECKPOINT)
    print(f"best val acc {best_acc:.3f} at epoch {best_epoch}")
    print(f"test acc {test_acc:.3f}")
    print(report)
    print(f"checkpoint {CHECKPOINT}")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _, eval_tf = make_transforms()
    image = Image.open(args.image)
    batch = eval_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]
    pred = int(probs.argmax())
    print(f"{idx_to_class[pred]}  {probs[pred].item():.3f}")
    for index in range(len(idx_to_class)):
        print(f"  {idx_to_class[index]}: {probs[index].item():.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run ResNet-18 empty/occupied classifier")
    parser.add_argument("--image", type=Path, help="Classify this image with the saved checkpoint")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.image:
        predict(args)
    else:
        train(args)
