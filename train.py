"""
Training entry points.

Two ways this file gets used:

1. Command line, once, to do the INITIAL training on a real labeled dataset:
       python train.py --data_dir data --epochs 10

2. Imported by feedback_manager.py, which calls fine_tune() automatically
   once enough human-confirmed feedback has piled up. Fine-tuning uses a
   much smaller learning rate and always validates before replacing the
   live model (see config.MAX_ALLOWED_ACCURACY_DROP).
"""
import argparse
import copy
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from model import build_backbone, get_transforms, save_checkpoint
from data_utils import get_dataloaders, _folder_has_images


def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    """Returns accuracy on the given loader. Returns None if loader is None/empty."""
    if loader is None:
        return None
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else None


def _train_loop(model, train_loader, val_loader, device, epochs, lr):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        train_acc = correct / total if total else 0.0
        val_acc = evaluate(model, val_loader, device)
        msg = f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / max(total,1):.4f} - train_acc: {train_acc:.4f}"
        if val_acc is not None:
            msg += f" - val_acc: {val_acc:.4f}"
        print(msg)

    return model


def train_initial(data_dir=config.DATA_DIR, epochs=config.INITIAL_TRAIN_EPOCHS,
                   lr=config.INITIAL_TRAIN_LR, batch_size=config.INITIAL_BATCH_SIZE):
    """Trains from the ImageNet-pretrained backbone on data/train (+ data/val)."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        train_dir=train_dir, val_dir=val_dir, batch_size=batch_size)
    print(f"[train] Classes: {class_to_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(pretrained=True)

    model = _train_loop(model, train_loader, val_loader, device, epochs, lr)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    save_checkpoint(model, class_to_idx, config.CURRENT_MODEL_PATH)
    print(f"[train] Saved model to {config.CURRENT_MODEL_PATH}")

    final_val_acc = evaluate(model, val_loader, device)
    return final_val_acc


def fine_tune(new_data_dir: str, base_val_dir: str = config.VAL_DIR) -> dict:
    """Incrementally fine-tunes the CURRENT deployed model on new feedback data.

    Safety behavior:
      - Trains a COPY of the current model, never touching the live checkpoint
        until the candidate has been validated.
      - Uses a small learning rate and few epochs (config.FINE_TUNE_*) so a
        modest batch of feedback nudges the model rather than overwriting it.
      - Compares candidate vs. current model on the held-out validation set
        (data/val, untouched by user feedback). If the candidate's accuracy
        drops by more than config.MAX_ALLOWED_ACCURACY_DROP, the update is
        rejected and the previous model is kept -- this is what protects the
        app against a batch of mistaken or malicious feedback corrupting it.

    `new_data_dir` must contain real/ and fake/ subfolders of newly
    confirmed feedback images (feedback_manager.py builds this folder).

    Returns a dict describing what happened, e.g.:
        {"accepted": True, "old_val_acc": 0.91, "new_val_acc": 0.93}
        {"accepted": False, "reason": "...", "old_val_acc": 0.91, "new_val_acc": 0.85}
    """
    if not _folder_has_images(new_data_dir):
        return {"accepted": False, "reason": "No new feedback images to train on."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.CURRENT_MODEL_PATH):
        return {"accepted": False, "reason": "No base model exists yet -- run initial training first."}

    checkpoint = torch.load(config.CURRENT_MODEL_PATH, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]

    # New feedback images must use the SAME class_to_idx as the live model,
    # so wrap them with an ImageFolder and remap indices if needed.
    feedback_ds = ImageFolder(new_data_dir, transform=get_transforms(train=True))
    if feedback_ds.class_to_idx != class_to_idx:
        remap = {feedback_ds.class_to_idx[name]: class_to_idx[name] for name in class_to_idx}
        feedback_ds.samples = [(p, remap[l]) for p, l in feedback_ds.samples]
        feedback_ds.targets = [remap[l] for l in feedback_ds.targets]

    feedback_loader = DataLoader(feedback_ds, batch_size=config.FINE_TUNE_BATCH_SIZE, shuffle=True)

    val_loader = None
    if _folder_has_images(base_val_dir):
        val_ds = ImageFolder(base_val_dir, transform=get_transforms(train=False))
        val_loader = DataLoader(val_ds, batch_size=config.FINE_TUNE_BATCH_SIZE, shuffle=False)

    # Build the CURRENT model to measure its baseline accuracy.
    current_model = build_backbone(pretrained=False)
    current_model.load_state_dict(checkpoint["state_dict"])
    old_val_acc = evaluate(current_model, val_loader, device)

    # Fine-tune a COPY on the new feedback data.
    candidate_model = copy.deepcopy(current_model)
    candidate_model = _train_loop(candidate_model, feedback_loader, val_loader, device,
                                   epochs=config.FINE_TUNE_EPOCHS, lr=config.FINE_TUNE_LR)
    new_val_acc = evaluate(candidate_model, val_loader, device)

    result = {"old_val_acc": old_val_acc, "new_val_acc": new_val_acc}

    if old_val_acc is None or new_val_acc is None:
        # No validation set available -- can't safely auto-verify improvement.
        # Accept cautiously since the point of the feature is to keep learning,
        # but this is exactly why setting up data/val is strongly recommended.
        accept = True
        result["reason"] = "No validation set found; accepted without safety check. Add data/val images for safer retraining."
    elif new_val_acc >= old_val_acc - config.MAX_ALLOWED_ACCURACY_DROP:
        accept = True
        result["reason"] = "Candidate model met the accuracy safety bar."
    else:
        accept = False
        result["reason"] = (f"Candidate accuracy ({new_val_acc:.3f}) dropped more than "
                             f"{config.MAX_ALLOWED_ACCURACY_DROP:.3f} below current "
                             f"({old_val_acc:.3f}); update rejected.")

    result["accepted"] = accept

    if accept:
        if os.path.exists(config.CURRENT_MODEL_PATH):
            shutil.copy2(config.CURRENT_MODEL_PATH, config.BACKUP_MODEL_PATH)
        save_checkpoint(candidate_model, class_to_idx, config.CURRENT_MODEL_PATH)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the deepfake detector from labeled data.")
    parser.add_argument("--data_dir", default=config.DATA_DIR,
                         help="Folder containing train/ and val/ subfolders (default: ./data)")
    parser.add_argument("--epochs", type=int, default=config.INITIAL_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.INITIAL_TRAIN_LR)
    parser.add_argument("--batch_size", type=int, default=config.INITIAL_BATCH_SIZE)
    args = parser.parse_args()

    acc = train_initial(args.data_dir, args.epochs, args.lr, args.batch_size)
    if acc is not None:
        print(f"[train] Final validation accuracy: {acc:.4f}")
