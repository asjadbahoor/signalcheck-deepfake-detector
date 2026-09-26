"""
Dataset helpers. Expects data organized as:

    data/train/real/*.jpg
    data/train/fake/*.jpg
    data/val/real/*.jpg
    data/val/fake/*.jpg

This is the same layout used by torchvision.datasets.ImageFolder, so that
class is used directly rather than reinventing it.
"""
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from model import get_transforms


def _folder_has_images(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path) and len(os.listdir(cls_path)) > 0:
            return True
    return False


def get_dataloaders(train_dir: str = config.TRAIN_DIR,
                     val_dir: str = config.VAL_DIR,
                     batch_size: int = config.INITIAL_BATCH_SIZE,
                     num_workers: int = 2):
    """Builds train/val DataLoaders. Returns (train_loader, val_loader, class_to_idx).

    Raises a clear error if the expected folders are empty, rather than
    silently training on nothing.
    """
    if not _folder_has_images(train_dir):
        raise FileNotFoundError(
            f"No training images found under {train_dir}.\n"
            f"Expected subfolders like:\n"
            f"  {train_dir}/real/*.jpg\n"
            f"  {train_dir}/fake/*.jpg\n"
            f"See README.md for links to public deepfake datasets."
        )

    train_ds = ImageFolder(train_dir, transform=get_transforms(train=True))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)

    val_loader = None
    if _folder_has_images(val_dir):
        val_ds = ImageFolder(val_dir, transform=get_transforms(train=False))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

    return train_loader, val_loader, train_ds.class_to_idx
