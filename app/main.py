"""
NeuroScan — FastAPI Backend (Hybrid ViT)
=========================================
Serves the NeuroScan UI and exposes a /predict endpoint that
loads your trained hybrid_vit .pth model and classifies brain MRI scans.

Architecture: hybrid_vit — ResNet-50 CNN + 6-layer Transformer encoder
Achieves ~99.21% validation accuracy on the 4-class brain tumor dataset.

Usage:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
  Then open http://localhost:8000
"""

import io
import os
import time
import random
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG
from model_loader import load_model

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("neuroscan")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroScan — Hybrid ViT", version="2.0.0")

# Serve static assets (css, js, images) if the folder exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Model bootstrap ────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL      = None
DEMO_MODE  = False

CLASS_NAMES = CONFIG["class_names"]   # ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

PREPROCESS = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


@app.on_event("startup")
async def startup():
    global MODEL, DEMO_MODE
    model_path = Path(CONFIG["model_path"])

    if not model_path.exists():
        log.warning(f"Model file not found at '{model_path}'. Running in DEMO mode.")
        DEMO_MODE = True
        return

    try:
        MODEL = load_model(
            architecture="hybrid_vit",
            num_classes=len(CLASS_NAMES),
            checkpoint_path=str(model_path),
            device=DEVICE,
        )
        MODEL.eval()
        log.info(f"✓ Hybrid ViT loaded from {model_path} on {DEVICE}")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.warning("Falling back to DEMO mode.")
        DEMO_MODE = True


# ── HTML ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in templates/")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"       : "ok",
        "demo_mode"    : DEMO_MODE,
        "architecture" : "hybrid_vit",
        "device"       : str(DEVICE),
        "model_path"   : CONFIG["model_path"],
        "classes"      : CLASS_NAMES,
    }


# ── Predict ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate file type ─────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Upload a JPG, PNG, WEBP, or BMP image."
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(raw) > CONFIG["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {CONFIG['max_upload_mb']} MB."
        )

    # ── Load image ─────────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot read the uploaded file as an image.")

    # ── Demo mode ──────────────────────────────────────────────────────────
    if DEMO_MODE:
        probs = _random_probs(len(CLASS_NAMES))
        pred_idx = probs.index(max(probs))
        return _build_response(
            pred_class  = CLASS_NAMES[pred_idx],
            confidence  = max(probs),
            probs_dict  = dict(zip(CLASS_NAMES, probs)),
            demo_mode   = True,
            latency_ms  = 0,
        )

    # ── Real inference ─────────────────────────────────────────────────────
    try:
        t0     = time.perf_counter()
        tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().tolist()

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        pred_idx   = probs.index(max(probs))

        log.info(
            f"Prediction: {CLASS_NAMES[pred_idx]} "
            f"({max(probs)*100:.1f}%)  [{latency_ms} ms]"
        )

        return _build_response(
            pred_class  = CLASS_NAMES[pred_idx],
            confidence  = max(probs) * 100,
            probs_dict  = {c: p * 100 for c, p in zip(CLASS_NAMES, probs)},
            demo_mode   = False,
            latency_ms  = latency_ms,
        )

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ── Helpers ────────────────────────────────────────────────────────────────
def _build_response(pred_class, confidence, probs_dict, demo_mode, latency_ms):
    return JSONResponse({
        "class"        : pred_class,
        "confidence"   : round(confidence, 2),
        "probabilities": {k: round(v, 2) for k, v in probs_dict.items()},
        "demo_mode"    : demo_mode,
        "latency_ms"   : latency_ms,
        "device"       : str(DEVICE),
    })


def _random_probs(n: int):
    """Generate plausible-looking random softmax probabilities."""
    raw = [random.uniform(0.02, 1.0) for _ in range(n)]
    total = sum(raw)
    return [round(r / total * 100, 2) for r in raw]
