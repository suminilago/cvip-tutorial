import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (
            x.shape[0],
            *([1] * (x.ndim - 1)),
        )

        random_tensor = (
            keep_prob
            + torch.rand(
                shape,
                dtype=x.dtype,
                device=x.device,
            )
        )

        random_tensor.floor_()

        return (
            x
            / keep_prob
            * random_tensor
        )


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=4,
        kernel_size=7,
        layer_scale_init_value=1e-6,
        drop_path=0.0,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        self.norm = nn.LayerNorm(dim)

        hidden_dim = dim * expansion_ratio

        self.pwconv1 = nn.Linear(
            dim,
            hidden_dim,
        )

        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(
            hidden_dim,
            dim,
        )

        self.gamma = nn.Parameter(
            layer_scale_init_value
            * torch.ones(dim)
        )

        self.drop_path = DropPath(
            drop_path
        )

    def forward(self, x):
        residual = x

        x = self.dwconv(x)

        x = x.permute(
            0, 2, 3, 1
        )

        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # LayerScale
        x = self.gamma * x

        x = x.permute(
            0, 3, 1, 2
        )

        # DropPath + Residual
        x = (
            residual
            + self.drop_path(x)
        )

        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        num_classes=30,
        depths=(2, 2, 3, 2),
        dims=(64, 128, 256, 512),
        kernel_size=7,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.1,
    ):
        super().__init__()

        total_blocks = sum(depths)

        drop_path_rates = torch.linspace(
            0,
            drop_path_rate,
            total_blocks,
        ).tolist()

        block_idx = 0

        # Patchify Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
            ),
        )

        # Stage 1
        stage1 = []

        for _ in range(depths[0]):
            stage1.append(
                ConvNeXtBlock(
                    dim=dims[0],
                    kernel_size=kernel_size,
                    layer_scale_init_value=(
                        layer_scale_init_value
                    ),
                    drop_path=(
                        drop_path_rates[
                            block_idx
                        ]
                    ),
                )
            )

            block_idx += 1

        self.stage1 = nn.Sequential(
            *stage1
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
        stage2 = []

        for _ in range(depths[1]):
            stage2.append(
                ConvNeXtBlock(
                    dim=dims[1],
                    kernel_size=kernel_size,
                    layer_scale_init_value=(
                        layer_scale_init_value
                    ),
                    drop_path=(
                        drop_path_rates[
                            block_idx
                        ]
                    ),
                )
            )

            block_idx += 1

        self.stage2 = nn.Sequential(
            *stage2
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
        stage3 = []

        for _ in range(depths[2]):
            stage3.append(
                ConvNeXtBlock(
                    dim=dims[2],
                    kernel_size=kernel_size,
                    layer_scale_init_value=(
                        layer_scale_init_value
                    ),
                    drop_path=(
                        drop_path_rates[
                            block_idx
                        ]
                    ),
                )
            )

            block_idx += 1

        self.stage3 = nn.Sequential(
            *stage3
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
        stage4 = []

        for _ in range(depths[3]):
            stage4.append(
                ConvNeXtBlock(
                    dim=dims[3],
                    kernel_size=kernel_size,
                    layer_scale_init_value=(
                        layer_scale_init_value
                    ),
                    drop_path=(
                        drop_path_rates[
                            block_idx
                        ]
                    ),
                )
            )

            block_idx += 1

        self.stage4 = nn.Sequential(
            *stage4
        )

        self.norm = nn.LayerNorm(
            dims[-1]
        )

        self.head = nn.Linear(
            dims[-1],
            num_classes,
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)

        x = self.downsample1(x)
        x = self.stage2(x)

        x = self.downsample2(x)
        x = self.stage3(x)

        x = self.downsample3(x)
        x = self.stage4(x)

        x = x.mean(
            dim=[2, 3]
        )

        x = self.norm(x)

        x = self.head(x)

        return x


if __name__ == "__main__":
    model = ConvNeXt(
        num_classes=30,
        depths=(2, 2, 3, 2),
        dims=(64, 128, 256, 512),
        kernel_size=3,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.1,
    )

    x = torch.randn(
        2,
        3,
        128,
        128,
    )

    y = model(x)

    print(
        "Input shape :",
        x.shape,
    )

    print(
        "Output shape:",
        y.shape,
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{total_params:,}",
    )