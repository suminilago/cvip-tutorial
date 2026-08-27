import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4, kernel_size=7):
        super().__init__()

        # 1. Depthwise Convolution
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        # 2. Layer Normalization
        self.norm = nn.LayerNorm(dim)

        # 3. Inverted Bottleneck
        hidden_dim = dim * expansion_ratio

        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x

        # Depthwise Conv
        x = self.dwconv(x)

        # (B, C, H, W) -> (B, H, W, C)
        # LayerNorm과 Linear를 channel dimension에 적용하기 위해 변경
        x = x.permute(0, 2, 3, 1)

        # LayerNorm
        x = self.norm(x)

        # 1x1 Conv 역할
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual Connection
        x = x + residual

        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        num_classes=30,
        depths=(2, 2, 3, 2),
        dims=(64, 128, 256, 512),
        kernel_size=7,
    ):
        super().__init__()

        # Patchify Stem
        # 128x128 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
            ),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=dims[0],
                    kernel_size=kernel_size,
                )
                for _ in range(depths[0])
            ]
        )

        # Downsample 1
        self.downsample1 = nn.Sequential(
            nn.Conv2d(
                dims[0],
                dims[1],
                kernel_size=2,
                stride=2,
            )
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=dims[1],
                    kernel_size=kernel_size,
                )
                for _ in range(depths[1])
            ]
        )

        # Downsample 2
        self.downsample2 = nn.Sequential(
            nn.Conv2d(
                dims[1],
                dims[2],
                kernel_size=2,
                stride=2,
            )
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=dims[2],
                    kernel_size=kernel_size,
                )
                for _ in range(depths[2])
            ]
        )

        # Downsample 3
        self.downsample3 = nn.Sequential(
            nn.Conv2d(
                dims[2],
                dims[3],
                kernel_size=2,
                stride=2,
            )
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=dims[3],
                    kernel_size=kernel_size,
                )
                for _ in range(depths[3])
            ]
        )

        # Final LayerNorm
        self.norm = nn.LayerNorm(dims[-1])

        # Classification Head
        self.head = nn.Linear(
            dims[-1],
            num_classes,
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.downsample1(x)
        x = self.stage2(x)

        # Stage 3
        x = self.downsample2(x)
        x = self.stage3(x)

        # Stage 4
        x = self.downsample3(x)
        x = self.stage4(x)

        # Global Average Pooling
        x = x.mean(dim=[2, 3])

        # LayerNorm
        x = self.norm(x)

        # Classification
        x = self.head(x)

        return x


if __name__ == "__main__":
    model = ConvNeXt(
        num_classes=30,
        depths=(2, 2, 3, 2),
        dims=(64, 128, 256, 512),
        kernel_size=7,
    )

    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    print(model)
    print()
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {total_params:,}")