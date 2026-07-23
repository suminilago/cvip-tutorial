import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidPoolingModule(nn.Module):
    """
    여러 크기의 adaptive average pooling으로
    지역 및 전역 문맥 정보를 수집합니다.
    """

    def __init__(
        self,
        in_channels: int,
        pool_sizes: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()

        if in_channels % len(pool_sizes) != 0:
            raise ValueError(
                "in_channels는 pool_sizes 개수로 나누어져야 합니다."
            )

        reduced_channels = in_channels // len(pool_sizes)

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(
                        in_channels,
                        reduced_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        pyramid_features = [x]

        for stage in self.stages:
            pooled = stage(x)

            upsampled = F.interpolate(
                pooled,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            pyramid_features.append(upsampled)

        return torch.cat(pyramid_features, dim=1)


class PSPNet(nn.Module):
    """
    ISBI 이진 segmentation 실습용 경량 PSPNet.

    원 논문의 ResNet backbone 대신,
    직접 구현 가능한 간단한 CNN encoder를 사용합니다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels,
                base_channels,
            ),
            ConvBlock(
                base_channels,
                base_channels * 2,
                stride=2,
            ),
            ConvBlock(
                base_channels * 2,
                base_channels * 4,
                stride=2,
            ),
            ConvBlock(
                base_channels * 4,
                base_channels * 8,
                stride=2,
            ),
            ConvBlock(
                base_channels * 8,
                base_channels * 8,
            ),
        )

        encoder_channels = base_channels * 8

        self.pyramid_pooling = PyramidPoolingModule(
            in_channels=encoder_channels,
            pool_sizes=(1, 2, 3, 6),
        )

        # 원본 feature 256채널 +
        # pooling feature 64채널 × 4 = 총 512채널
        pyramid_output_channels = encoder_channels * 2

        self.decoder = nn.Sequential(
            ConvBlock(
                pyramid_output_channels,
                base_channels * 4,
            ),
            nn.Dropout2d(p=0.1),
            ConvBlock(
                base_channels * 4,
                base_channels * 2,
            ),
            nn.Conv2d(
                base_channels * 2,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.encoder(x)
        x = self.pyramid_pooling(x)
        x = self.decoder(x)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


def main():
    model = PSPNet(
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

    # BatchNorm 때문에 테스트 시 eval 모드 사용
    model.eval()

    with torch.no_grad():
        sample_output = model(sample_input)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 60)
    print("PSPNet 모델 확인")
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