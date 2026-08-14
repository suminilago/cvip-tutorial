from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IMAGE_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((144, 144)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_datasets():
    train_dataset = datasets.ImageFolder(
        root=DATA_DIR / "train",
        transform=get_train_transform(),
    )

    val_dataset = datasets.ImageFolder(
        root=DATA_DIR / "val",
        transform=get_eval_transform(),
    )

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "test",
        transform=get_eval_transform(),
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train과 validation의 클래스 순서가 다릅니다.")

    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Train과 test의 클래스 순서가 다릅니다.")

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size=32,
    num_workers=0,
):
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader