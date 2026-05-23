# Brain Tumor Hybrid ViT Project

End-to-end brain tumor MRI classification project using a Hybrid CNN + Vision Transformer architecture.

This repository has two main parts:
- `train/`: model training, validation, checkpointing, and prediction utilities
- `app/`: FastAPI web app for serving predictions from a trained checkpoint

## Project Overview

The model architecture is **`hybrid_vit`**:
- ResNet-50 backbone (up to layer3) extracts local visual features
- Feature map is converted to token sequence
- 6-layer Transformer encoder models global context
- CLS token is used for final classification

Target classes (training side) are typically:
- `glioma`
- `meningioma`
- `no_tumor`
- `pituitary`

## Repository Layout

```text
Brain-tumor-hybrid-only/
├── train/
│   ├── train.py
│   ├── predict.py
│   ├── models/
│   ├── utils/
│   ├── requirements.txt
│   └── README.md
└── app/
    ├── main.py
    ├── model_loader.py
    ├── config.py
    ├── templates/
    ├── requirements.txt
    └── README.md
```

## 1. Training Setup

From the repository root:

```bash
cd train
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Expected dataset layout

Training code expects this structure under `train/data/`:

```text
train/data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

## 2. Train the Model

Basic training:

```bash
python train.py
```

Recommended (MixUp + 2-stage training):

```bash
python train.py --epochs 40 --freeze_epochs 5 --use_mixup
```

Main output files (default `train/outputs/`):
- `best_model.pth`
- `training_curves.png`
- `confusion_matrix.png`

## 3. Run CLI Inference (Optional)

Single image:

```bash
python predict.py --image path/to/mri.jpg --checkpoint outputs/best_model.pth
```

With attention visualization:

```bash
python predict.py \
  --image path/to/mri.jpg \
  --checkpoint outputs/best_model.pth \
  --show_attention
```

Prediction plots are saved by default to `train/outputs/predictions/`.

## 4. Deploy the FastAPI App

Open a new shell and set up the app environment:

```bash
cd app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the trained checkpoint from training outputs into `app/` (or point config/env var to it):

```bash
cp ../train/outputs/best_model.pth ./hyb_model.pth
```

Start the API server:

```bash
uvicorn main:app --reload --port 8000
```

Open:
- `http://localhost:8000` (web UI)
- `http://localhost:8000/health` (health endpoint)

## 5. Configuration Notes

Model path and app settings are in `app/config.py`.

You can also override checkpoint path via environment variable:

```bash
MODEL_PATH=/absolute/path/to/best_model.pth uvicorn main:app --reload --port 8000
```

Important:
- Keep class order in app config aligned with the training checkpoint class order.
- If class naming differs (for example `no_tumor` vs `no-tumor`), update `app/config.py` to match your trained class labels.

## 6. API Endpoints

- `GET /`: serves the frontend
- `GET /health`: reports status, device, and model mode
- `POST /predict`: image classification endpoint

`/predict` accepts image uploads (`jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`) and returns class probabilities.

## Troubleshooting

- App starts in demo mode:
  - Check that checkpoint exists at `app/config.py -> model_path`.
- CUDA not used:
  - Confirm GPU-enabled PyTorch install and CUDA availability.
- Class mismatch in outputs:
  - Ensure class names/order in `app/config.py` match training labels saved in checkpoint.

## Module Docs

More detailed component docs:
- `train/README.md`
- `app/README.md`

## Disclaimer

This project is for research/educational use and is **not** a clinical diagnostic tool.
