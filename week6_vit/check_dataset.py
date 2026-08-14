from dataset import get_dataloaders, get_datasets


def main():
    train_dataset, val_dataset, test_dataset = get_datasets()

    print("=" * 70)
    print("ViT 데이터셋 확인")
    print("=" * 70)

    print(f"Train 이미지 수: {len(train_dataset)}")
    print(f"Validation 이미지 수: {len(val_dataset)}")
    print(f"Test 이미지 수: {len(test_dataset)}")
    print(f"클래스 수: {len(train_dataset.classes)}")

    print("\n클래스 목록:")
    for index, class_name in enumerate(train_dataset.classes):
        print(f"{index:02d}: {class_name}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("\n" + "=" * 70)
    print("DataLoader 배치 확인")
    print("=" * 70)
    print(f"이미지 배치 크기: {images.shape}")
    print(f"라벨 배치 크기: {labels.shape}")
    print(f"라벨 예시: {labels[:10]}")
    print(f"이미지 dtype: {images.dtype}")
    print(f"라벨 dtype: {labels.dtype}")

    expected_shape = (32, 3, 128, 128)

    if tuple(images.shape) == expected_shape:
        print("\n배치 크기 정상")
    else:
        print(
            f"\n예상 크기 {expected_shape}와 "
            f"실제 크기 {tuple(images.shape)}가 다릅니다."
        )

    print("\nTrain 배치 수:", len(train_loader))
    print("Validation 배치 수:", len(val_loader))
    print("Test 배치 수:", len(test_loader))


if __name__ == "__main__":
    main()