import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Encoder에서 사용하는 기본 convolution 블록입니다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationBlock(nn.Module):
    """
    각각의 pyramid feature를 segmentation용 feature로 변환합니다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNSegmentation(nn.Module):
    """
    ISBI 2012 이진 segmentation용 경량 FPN.

    입력:
        [batch, 1, height, width]

    출력:
        [batch, 1, height, width]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        pyramid_channels: int = 128,
        segmentation_channels: int = 64,
    ):
        super().__init__()

        # Bottom-up encoder
        self.stage1 = ConvBlock(
            in_channels,
            base_channels,
            stride=1,
        )

        self.stage2 = ConvBlock(
            base_channels,
            base_channels * 2,
            stride=2,
        )

        self.stage3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            stride=2,
        )

        self.stage4 = ConvBlock(
            base_channels * 4,
            base_channels * 8,
            stride=2,
        )

        self.stage5 = ConvBlock(
            base_channels * 8,
            base_channels * 16,
            stride=2,
        )

        # Lateral 1×1 convolutions
        self.lateral2 = nn.Conv2d(
            base_channels * 2,
            pyramid_channels,
            kernel_size=1,
        )

        self.lateral3 = nn.Conv2d(
            base_channels * 4,
            pyramid_channels,
            kernel_size=1,
        )

        self.lateral4 = nn.Conv2d(
            base_channels * 8,
            pyramid_channels,
            kernel_size=1,
        )

        self.lateral5 = nn.Conv2d(
            base_channels * 16,
            pyramid_channels,
            kernel_size=1,
        )

        # 합친 feature를 부드럽게 만드는 3×3 convolution
        self.smooth2 = nn.Conv2d(
            pyramid_channels,
            pyramid_channels,
            kernel_size=3,
            padding=1,
        )

        self.smooth3 = nn.Conv2d(
            pyramid_channels,
            pyramid_channels,
            kernel_size=3,
            padding=1,
        )

        self.smooth4 = nn.Conv2d(
            pyramid_channels,
            pyramid_channels,
            kernel_size=3,
            padding=1,
        )

        self.smooth5 = nn.Conv2d(
            pyramid_channels,
            pyramid_channels,
            kernel_size=3,
            padding=1,
        )

        self.segmentation2 = SegmentationBlock(
            pyramid_channels,
            segmentation_channels,
        )

        self.segmentation3 = SegmentationBlock(
            pyramid_channels,
            segmentation_channels,
        )

        self.segmentation4 = SegmentationBlock(
            pyramid_channels,
            segmentation_channels,
        )

        self.segmentation5 = SegmentationBlock(
            pyramid_channels,
            segmentation_channels,
        )

        combined_channels = segmentation_channels * 4

        self.output_head = nn.Sequential(
            nn.Conv2d(
                combined_channels,
                segmentation_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(segmentation_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(
                segmentation_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    @staticmethod
    def upsample_add(
        higher_feature: torch.Tensor,
        lateral_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        상위 pyramid feature를 lateral feature 크기로 키운 뒤 더합니다.
        """

        higher_feature = F.interpolate(
            higher_feature,
            size=lateral_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return higher_feature + lateral_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Bottom-up feature 추출
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        # Top-down pathway
        p5 = self.lateral5(c5)

        p4 = self.upsample_add(
            p5,
            self.lateral4(c4),
        )

        p3 = self.upsample_add(
            p4,
            self.lateral3(c3),
        )

        p2 = self.upsample_add(
            p3,
            self.lateral2(c2),
        )

        # Aliasing 감소를 위한 smoothing
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        # 각 pyramid level을 segmentation feature로 변환
        s2 = self.segmentation2(p2)
        s3 = self.segmentation3(p3)
        s4 = self.segmentation4(p4)
        s5 = self.segmentation5(p5)

        target_size = s2.shape[-2:]

        s3 = F.interpolate(
            s3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        s4 = F.interpolate(
            s4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        s5 = F.interpolate(
            s5,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # 다중 해상도 feature 결합
        x = torch.cat(
            [s2, s3, s4, s5],
            dim=1,
        )

        x = self.output_head(x)

        # 원본 입력 크기로 복원
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


def main():
    model = FPNSegmentation(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        pyramid_channels=128,
        segmentation_channels=64,
    )

    sample_input = torch.randn(
        2,
        1,
        256,
        256,
    )

    model.eval()

    with torch.no_grad():
        sample_output = model(sample_input)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 60)
    print("FPN 모델 확인")
    print("=" * 60)

    print(f"입력 shape: {sample_input.shape}")
    print(f"출력 shape: {sample_output.shape}")
    print(f"전체 파라미터 수: {num_parameters:,}")

    if sample_input.shape == sample_output.shape:
        print("입력과 출력 shape가 같습니다.")
    else:
        print("입력과 출력 shape가 다릅니다.")


if __name__ == "__main__":
    main()