from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "unmodified-data"

TRAIN_IMG_DIR = DATA_DIR / "train" / "imgs"
TRAIN_LABEL_DIR = DATA_DIR / "train" / "labels"


class ISBI2012Dataset(Dataset):
    """
    ISBI 2012 binary semantic segmentation dataset.

    이미지:
        grayscale, 0~255

    마스크:
        background = 0
        foreground = 255
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: Optional[Tuple[int, int]] = (256, 256),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"이미지 폴더를 찾을 수 없습니다: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"마스크 폴더를 찾을 수 없습니다: {self.mask_dir}"
            )

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise RuntimeError("이미지 파일이 없습니다.")

        if len(self.mask_paths) == 0:
            raise RuntimeError("마스크 파일이 없습니다.")

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                "이미지 수와 마스크 수가 다릅니다. "
                f"images={len(self.image_paths)}, "
                f"masks={len(self.mask_paths)}"
            )

        for image_path, mask_path in zip(
            self.image_paths,
            self.mask_paths,
        ):
            if image_path.name != mask_path.name:
                raise RuntimeError(
                    "이미지와 마스크 파일명이 일치하지 않습니다.\n"
                    f"image: {image_path.name}\n"
                    f"mask: {mask_path.name}"
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            image = TF.resize(
                image,
                self.image_size,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

            mask = TF.resize(
                mask,
                self.image_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image_tensor = TF.to_tensor(image)
        mask_tensor = TF.to_tensor(mask)

        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


def main():
    dataset = ISBI2012Dataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_LABEL_DIR,
        image_size=(256, 256),
    )

    image, mask = dataset[0]

    print("=" * 60)
    print("PyTorch Dataset 확인")
    print("=" * 60)

    print(f"데이터셋 길이: {len(dataset)}")
    print(f"이미지 shape: {image.shape}")
    print(f"마스크 shape: {mask.shape}")
    print(f"이미지 dtype: {image.dtype}")
    print(f"마스크 dtype: {mask.dtype}")
    print(f"이미지 최솟값: {image.min().item():.4f}")
    print(f"이미지 최댓값: {image.max().item():.4f}")
    print(f"마스크 고유값: {torch.unique(mask)}")


if __name__ == "__main__":
    main()