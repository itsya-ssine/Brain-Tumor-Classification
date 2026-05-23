"""
Vision Transformer Model for Brain Tumor MRI Classification
=============================================================
Architecture:

  hybrid_vit  — CNN (ResNet50) feature extractor + ViT encoder
                Combines local CNN features with global attention
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────
# Hybrid CNN + ViT
# ─────────────────────────────────────────────────────────────
class ConvPatchEmbed(nn.Module):
    """
    ResNet-50 stem used as a convolutional patch embedder.
    Converts (B, 3, H, W)  →  (B, N, embed_dim) token sequence.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        # Keep everything up to layer3 (stride-16 feature map)
        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,       # → (B, 1024, H/16, W/16)
        )
        self.proj = nn.Conv2d(1024, embed_dim, kernel_size=1)  # channel projection

    def forward(self, x):
        feat = self.cnn(x)           # (B, 1024, H/16, W/16)
        feat = self.proj(feat)       # (B, embed_dim, H/16, W/16)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, H*W, embed_dim)
        return tokens, H, W


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder (6 layers, 8 heads)."""
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.encoder(x))


class HybridViTClassifier(nn.Module):
    """
    Hybrid CNN + Vision Transformer.

    Architecture:
        ResNet-50 (up to layer3)  →  Conv patch tokens
                                  →  Learnable [CLS] token
                                  →  Positional embeddings
                                  →  6-layer Transformer encoder
                                  →  [CLS] → classifier head

    Why hybrid?
    - CNN extracts low-level local features (edges, textures) that ViT
      struggles to learn from scratch on small datasets.
    - Transformer then models global dependencies across the CNN patches.
    - Best of both worlds; typically outperforms pure ViT when data is limited.

    Args:
        num_classes (int): Number of output classes. Default 4.
        embed_dim   (int): Token embedding dimension. Default 768.
        depth       (int): Number of transformer layers. Default 6.
        num_heads   (int): Number of attention heads. Default 8.
        dropout     (float): Dropout probability. Default 0.2.
        img_size    (int): Input image size. Default 224.
    """
    def __init__(self, num_classes=4, embed_dim=768, depth=6,
                 num_heads=8, dropout=0.2, img_size=224):
        super().__init__()

        self.patch_embed = ConvPatchEmbed(embed_dim=embed_dim)

        # Spatial grid after ResNet stride-16
        n_patches = (img_size // 16) ** 2      # 14×14 = 196 for img_size=224
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        tokens, H, W = self.patch_embed(x)           # (B, N, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)      # (B, N+1, embed_dim)
        tokens = self.pos_drop(tokens + self.pos_embed)

        tokens = self.encoder(tokens)                 # (B, N+1, embed_dim)
        cls_out = tokens[:, 0]                        # (B, embed_dim)  — CLS token
        return self.head(cls_out)

    def freeze_cnn(self):
        """Freeze ResNet feature extractor, train only transformer + head."""
        for p in self.patch_embed.cnn.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        """Alias for freeze_cnn — freezes CNN feature extractor."""
        self.freeze_cnn()

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
def build_model(architecture: str = "hybrid_vit", num_classes: int = 4,
                pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Factory function — returns the HybridViTClassifier.

    Args:
        architecture (str): Must be 'hybrid_vit' (only supported option).
        num_classes  (int): Number of output classes.
        pretrained   (bool): Use pretrained ResNet-50 weights (always True for hybrid).
        **kwargs: Extra args forwarded to HybridViTClassifier constructor.

    Returns:
        HybridViTClassifier
    """
    arch = architecture.lower()
    if arch != "hybrid_vit":
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"This project only supports: hybrid_vit"
        )
    return HybridViTClassifier(num_classes=num_classes, **kwargs)


def model_summary(model: nn.Module, name: str = ""):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*52}")
    print(f"  Model: {name or model.__class__.__name__}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Frozen params:    {total-trainable:>12,}")
    print(f"{'='*52}\n")
