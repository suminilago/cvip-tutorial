import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    3x3 convolution을 두 번 적용하는 기본 블록
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
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


class DownBlock(nn.Module):
    """
    MaxPool로 크기를 절반으로 줄인 뒤 DoubleConv 적용
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    ConvTranspose2d로 크기를 두 배로 늘린 뒤
    encoder feature와 concatenate하고 DoubleConv 적용
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                "업샘플링 결과와 skip feature 크기가 다릅니다.\n"
                f"x shape: {x.shape}\n"
                f"skip shape: {skip.shape}"
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Binary semantic segmentation용 U-Net

    입력:
        [batch, 1, height, width]

    출력:
        [batch, 1, height, width]

    출력에는 sigmoid를 적용하지 않습니다.
    학습 시 BCEWithLogitsLoss를 사용합니다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        self.encoder1 = DoubleConv(
            in_channels,
            base_channels,
        )

        self.encoder2 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.encoder3 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.encoder4 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.bottleneck = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.decoder4 = UpBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
        )

        self.decoder3 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.decoder2 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.decoder1 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)

        x = self.bottleneck(skip4)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        return self.output_conv(x)


def main():
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    )

    sample_input = torch.randn(
        2,
        1,
        256,
        256,
    )

    with torch.no_grad():
        sample_output = model(sample_input)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 60)
    print("U-Net 모델 확인")
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