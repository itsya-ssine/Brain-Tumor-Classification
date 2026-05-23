"""
model_loader.py
===============
Loads the HybridViTClassifier from a .pth checkpoint.
Architecture: ResNet-50 CNN feature extractor + 6-layer Transformer encoder.

IMPORTANT: The class structure here must match the training code exactly,
including the TransformerEncoder wrapper, so that state_dict keys align.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ═══════════════════════════════════════════════════════════════════
# Hybrid CNN + ViT  (mirrors Brain-tumor-hybrid-only/models/vit_model.py)
# ═══════════════════════════════════════════════════════════════════

class ConvPatchEmbed(nn.Module):
    """
    ResNet-50 stem used as a convolutional patch embedder.
    Converts (B, 3, H, W) → (B, N, embed_dim) token sequence.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        r = tv_models.resnet50(weights=None)
        # Keep everything up to layer3 (stride-16 feature map)
        self.cnn = nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool,
            r.layer1, r.layer2, r.layer3,      # → (B, 1024, H/16, W/16)
        )
        self.proj = nn.Conv2d(1024, embed_dim, kernel_size=1)

    def forward(self, x):
        feat = self.cnn(x)                          # (B, 1024, H/16, W/16)
        feat = self.proj(feat)                      # (B, embed_dim, H/16, W/16)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, H*W, embed_dim)
        return tokens, H, W


class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder (6 layers, 8 heads).
    
    Kept as a named wrapper so that state_dict keys match the checkpoint:
      encoder.encoder.layers.*  and  encoder.norm.*
    """
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
                                  →  TransformerEncoder (6 layers, 8 heads)
                                  →  [CLS] → MLP classifier head

    State-dict key layout (must match checkpoint):
        patch_embed.cnn.*
        patch_embed.proj.*
        cls_token
        pos_embed
        pos_drop.*          (Dropout — no learned params)
        encoder.encoder.layers.*   ← TransformerEncoder wrapper
        encoder.norm.*             ← TransformerEncoder wrapper
        head.*
    """
    def __init__(self, num_classes=4, embed_dim=768, depth=6,
                 num_heads=8, dropout=0.2, img_size=224):
        super().__init__()

        self.patch_embed = ConvPatchEmbed(embed_dim=embed_dim)

        n_patches = (img_size // 16) ** 2      # 14×14 = 196 for img_size=224
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # Named wrapper — produces keys: encoder.encoder.layers.* / encoder.norm.*
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout,
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
        cls_out = tokens[:, 0]                        # (B, embed_dim)
        return self.head(cls_out)


# ═══════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════

def load_model(
    architecture: str,
    num_classes: int,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Build HybridViTClassifier and load weights from checkpoint.

    The checkpoint is expected to be a dict with key 'model_state_dict',
    as saved by the training script. A bare state_dict (OrderedDict)
    is also accepted for flexibility.

    Args:
        architecture   : Must be 'hybrid_vit'
        num_classes    : Number of output classes
        checkpoint_path: Path to the .pth file
        device         : torch device to load onto

    Returns:
        HybridViTClassifier in eval() mode
    """
    if architecture.lower() != "hybrid_vit":
        raise ValueError(
            f"This build only supports 'hybrid_vit'. Got: '{architecture}'"
        )

    model = HybridViTClassifier(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both formats: bare state_dict OR {model_state_dict: ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch   = checkpoint.get("epoch", "?")
        val_acc = checkpoint.get("val_acc", None)
        if val_acc is not None:
            print(f"  Checkpoint: epoch={epoch}, val_acc={val_acc:.2f}%")
        saved_classes = checkpoint.get("class_names")
        if saved_classes:
            print(f"  Saved class names: {saved_classes}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
