from __future__ import annotations

import io
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_PATH = os.getenv("MODEL_PATH", "runs/apple_logo2/weights/best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.5"))

app = FastAPI(title="Apple Logo Detector", version="1.0.0")


def _load_model() -> YOLO:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


model = _load_model()


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Use: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    results = model.predict(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    if not results:
        return JSONResponse({"detections": []})

    detections: List[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.squeeze().tolist()
            if len(xyxy) != 4:
                continue
            detections.append(
                {
                    "class": "apple_logo",
                    "confidence": float(b.conf.item()),
                    "bbox": [float(x) for x in xyxy],
                }
            )

    return JSONResponse({"detections": detections})
