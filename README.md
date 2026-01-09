<<<<<<< HEAD
# ml_ops
opda
=======
﻿# Apple Logo Detection - Submission

## Summary

I built a REST API service that detects the Apple logo (bitten apple, any color) and returns bounding boxes. The model is already trained and I use the current weights without retraining.

## Goal

The service accepts an image and returns detections of the Apple logo.

## Constraints

- Processing time: up to 10 seconds per image
- Hardware: 16 GB GPU (Google Colab T4 level)
- Formats: JPEG, PNG, BMP, WEBP

## Repository layout

```
apple_case/
  app/                 # REST API (FastAPI)
  scripts/             # validation script
  data/                # dataset (local)
  runs/                # training artifacts (local)
  models/              # model weights (local)
  Dockerfile
  requirements.txt
  README.md
```

## Data and preparation

I collected real Apple logo assets with transparent background in `true_apple/` and generated a synthetic dataset by compositing logos on varied backgrounds with different sizes, rotations, and colors. The dataset is in YOLO format and split 80/20 into train/val.

## Model

I use YOLOv8 (Ultralytics). Current weights:
- Local: `runs/apple_logo2/weights/best.pt`
- Public link: TODO (add link)

## Training results

Final validation metrics (epoch 50, `runs/apple_logo2/results.csv`):
- Precision: 0.9997
- Recall: 1.0000
- mAP@0.5: 0.9950
- mAP@0.5:0.95: 0.9797

## Run API (local)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Request example

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/image.jpg"
```

Response example:

```json
{
  "detections": [
    {
      "class": "apple_logo",
      "confidence": 0.97,
      "bbox": [x_min, y_min, x_max, y_max]
    }
  ]
}
```

## Run in Docker

1) Place weights into `models/best.pt`
2) Build and run:

```bash
docker build -t apple-logo-detector .
docker run -p 8000:8000 -v %cd%/models:/app/models apple-logo-detector
```

Service runs on port `8000`.

## Validation (F1-score, IoU=0.5)

```bash
python scripts/validate.py \
  --images data/splits/val/images \
  --labels data/splits/val/labels \
  --weights runs/apple_logo2/weights/best.pt \
  --conf 0.25 \
  --iou 0.5
```
