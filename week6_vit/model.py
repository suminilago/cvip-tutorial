import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size는 patch_size로 나누어떨어져야 합니다."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # 입력: [B, 3, 128, 128]
        x = self.projection(x)

        # [B, embed_dim, 8, 8]
        x = x.flatten(2)

        # [B, embed_dim, 64] -> [B, 64, embed_dim]
        x = x.transpose(1, 2)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        attention_output, _ = self.attention(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )

        return attention_output


class MLPBlock(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        mlp_dim=384,
        dropout=0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        mlp_dim=384,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        # Pre-Norm + Residual Connection
        x = x + self.attention(self.norm1(x))

        # Pre-Norm + Residual Connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=30,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_dim=384,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.patch_embedding.projection.weight,
            std=0.02,
        )

        if self.patch_embedding.projection.bias is not None:
            nn.init.zeros_(
                self.patch_embedding.projection.bias
            )

        nn.init.zeros_(self.classifier.bias)
        nn.init.trunc_normal_(
            self.classifier.weight,
            std=0.02,
        )

    def forward(self, x):
        batch_size = x.size(0)

        # [B, 64, 192]
        x = self.patch_embedding(x)

        # [1, 1, 192] -> [B, 1, 192]
        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # [B, 65, 192]
        x = torch.cat(
            [cls_tokens, x],
            dim=1,
        )

        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        x = self.norm(x)

        # CLS Token만 사용
        cls_output = x[:, 0]

        logits = self.classifier(cls_output)

        return logits


def build_vit(
    num_classes=30,
    depth=4,
):
    return VisionTransformer(
        image_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_dim=384,
        dropout=0.1,
    )


if __name__ == "__main__":
    model = build_vit(
        num_classes=30,
        depth=4,
    )

    dummy_input = torch.randn(
        4,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 70)
    print("Vision Transformer 모델 확인")
    print("=" * 70)
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"전체 파라미터 수: {total_parameters:,}")
    print(f"학습 파라미터 수: {trainable_parameters:,}")

    if tuple(output.shape) == (4, 30):
        print("\n모델 출력 크기 정상")
    else:
        print("\n모델 출력 크기 오류")