from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent

# week6_vit에서 사용했던 Plants 데이터셋 그대로 재사용
DATA_DIR = BASE_DIR.parent / "week6_vit" / "data"

IMAGE_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 기존 Baseline Augmentation
# ============================================================

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


# ============================================================
# 개선 실험용 Strong Augmentation
# ============================================================

def get_train_transform_strong():
    return transforms.Compose(
        [
            transforms.Resize((144, 144)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),

            # 색상/밝기/대비 변화에 대한 일반화 성능 개선
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# Validation / Test Transform
# ============================================================

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


# ============================================================
# 기존 Dataset
# ============================================================

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
        raise ValueError(
            "Train과 validation의 클래스 순서가 다릅니다."
        )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Train과 test의 클래스 순서가 다릅니다."
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# Strong Augmentation Dataset
# ============================================================

def get_datasets_strong_aug():
    train_dataset = datasets.ImageFolder(
        root=DATA_DIR / "train",
        transform=get_train_transform_strong(),
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
        raise ValueError(
            "Train과 validation의 클래스 순서가 다릅니다."
        )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Train과 test의 클래스 순서가 다릅니다."
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# 기존 DataLoader
# ============================================================

def get_dataloaders(
    batch_size=32,
    num_workers=0,
):
    train_dataset, val_dataset, test_dataset = (
        get_datasets()
    )

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

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Strong Augmentation DataLoader
# ============================================================

def get_dataloaders_strong_aug(
    batch_size=32,
    num_workers=0,
):
    train_dataset, val_dataset, test_dataset = (
        get_datasets_strong_aug()
    )

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

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Dataset Test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Baseline Dataset")
    print("=" * 70)

    train_dataset, val_dataset, test_dataset = (
        get_datasets()
    )

    print("Dataset directory:", DATA_DIR)
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print(
        "Number of classes:",
        len(train_dataset.classes),
    )

    print("Classes:")
    print(train_dataset.classes)

    train_loader, _, _ = get_dataloaders(
        batch_size=32
    )

    images, labels = next(
        iter(train_loader)
    )

    print()
    print(
        "Baseline batch image shape:",
        images.shape,
    )
    print(
        "Baseline batch label shape:",
        labels.shape,
    )

    print()
    print("=" * 70)
    print("Strong Augmentation Dataset")
    print("=" * 70)

    strong_train_loader, _, _ = (
        get_dataloaders_strong_aug(
            batch_size=32
        )
    )

    strong_images, strong_labels = next(
        iter(strong_train_loader)
    )

    print(
        "Strong batch image shape:",
        strong_images.shape,
    )

    print(
        "Strong batch label shape:",
        strong_labels.shape,
    )