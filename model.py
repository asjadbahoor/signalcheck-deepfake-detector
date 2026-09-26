"""
Model definition and inference helpers for the Deepfake Detector.

Uses an ImageNet-pretrained EfficientNet-B0 backbone (transfer learning)
with a small custom classifier head for the binary real/fake task. Transfer
learning is used because training a strong image classifier from random
weights needs far more data and compute than most people training a
deepfake detector from scratch will have available.
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import config


def build_backbone(pretrained: bool = True) -> nn.Module:
    """Builds an EfficientNet-B0 with a 2-class classification head.

    If pretrained ImageNet weights can't be downloaded (e.g. no internet
    access), falls back to random initialization with a clear warning --
    the app will still run, but accuracy will be much lower until trained.
    """
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
    except Exception as e:  # no internet, corrupted cache, etc.
        print(f"[model] Could not load pretrained weights ({e}). "
              f"Falling back to randomly initialized weights.")
        net = models.efficientnet_b0(weights=None)

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(128, len(config.CLASS_NAMES)),
    )
    return net


def get_transforms(train: bool = False) -> transforms.Compose:
    """Preprocessing pipeline. Light augmentation is applied only at train time."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats, matches the pretrained backbone
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


# Default label mapping used until a real checkpoint (trained via train.py)
# provides its own class_to_idx. This matches torchvision.ImageFolder's
# default behavior of sorting class folder names alphabetically
# ("fake" < "real"), so it lines up automatically with data/train/{fake,real}.
DEFAULT_IDX_TO_CLASS = {0: "fake", 1: "real"}


def save_checkpoint(model: nn.Module, class_to_idx: dict, path: str):
    """Saves weights together with the label mapping used at training time,
    so inference never has to guess which index means 'real' vs 'fake'."""
    torch.save({"state_dict": model.state_dict(), "class_to_idx": class_to_idx}, path)


class DeepfakeDetector:
    """Thin wrapper that owns the model, device placement, and inference logic."""

    def __init__(self, model_path: str = config.CURRENT_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.idx_to_class = dict(DEFAULT_IDX_TO_CLASS)
        self.model = self._load_or_init()
        self.model.to(self.device)
        self.model.eval()
        self.eval_transform = get_transforms(train=False)

    def _load_or_init(self) -> nn.Module:
        # Hosted instances may not have enough startup time to download the
        # optional ImageNet weights. Keep the full pretrained behavior for
        # local use, while allowing Render to opt out with an environment var.
        use_pretrained = os.getenv("DEEPFAKE_USE_PRETRAINED", "1").lower() not in {
            "0", "false", "no"
        }
        net = build_backbone(pretrained=use_pretrained)
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu")
                net.load_state_dict(checkpoint["state_dict"])
                class_to_idx = checkpoint.get("class_to_idx")
                if class_to_idx:
                    self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                print(f"[model] Loaded fine-tuned weights from {self.model_path} "
                      f"(labels: {self.idx_to_class})")
            except Exception as e:
                print(f"[model] Failed to load checkpoint ({e}); using base backbone.")
        else:
            print("[model] No trained checkpoint found yet -- using ImageNet backbone "
                  "only. Run train.py on a labeled dataset before relying on results.")
        return net

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """Runs inference on a single PIL image.

        Returns a dict: {label, confidence, probabilities: {real, fake}}
        """
        image = image.convert("RGB")
        tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        label = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)}

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def reload(self):
        """Re-reads the checkpoint from disk -- call after a retrain swaps the model file."""
        self.model = self._load_or_init()
        self.model.to(self.device)
        self.model.eval()
