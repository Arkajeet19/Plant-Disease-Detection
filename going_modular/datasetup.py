import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

NUM_WORKERS = os.cpu_count() or 0
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "New Plant Diseases Dataset(Augmented)"


def create_dataloaders(
    data_root: str | Path | None = None,
    train_transform: v2.Compose | None = None,
    test_transform: v2.Compose | None = None,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
):
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT

    train_data = datasets.ImageFolder(
        root=str(root / "train"),
        transform=train_transform,
    )
    test_data = datasets.ImageFolder(
        root=str(root / "valid"),
        transform=test_transform,
    )

    if train_data.class_to_idx != test_data.class_to_idx:
        raise ValueError("Train and valid folders must contain the same class folders.")

    class_names = train_data.classes

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_data_loader, test_data_loader, class_names



     




