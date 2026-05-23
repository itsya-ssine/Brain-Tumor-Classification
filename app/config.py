"""
config.py — NeuroScan configuration (Hybrid ViT)
=================================================
Edit this file to point to your trained hybrid_vit checkpoint.
"""

import os

CONFIG = {
    # ── Path to your saved .pth checkpoint ──────────────────────────────
    # Can also be set via environment variable:  MODEL_PATH=path/to/model.pth
    "model_path": os.getenv("MODEL_PATH", "hyb_model.pth"),

    # ── Architecture — fixed to hybrid_vit ───────────────────────────────
    "architecture": "hybrid_vit",

    # ── Class names — must match training order exactly ──────────────────
    # These match the Kaggle dataset folder names used during training:
    "class_names": ["glioma", "meningioma", "no-tumor", "pituitary"],

    # ── Image size fed to the model ──────────────────────────────────────
    "img_size": int(os.getenv("IMG_SIZE", "224")),

    # ── Max upload size in MB ────────────────────────────────────────────
    "max_upload_mb": int(os.getenv("MAX_UPLOAD_MB", "20")),
}
