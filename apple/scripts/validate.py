from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from ultralytics import YOLO


def yolo_to_xyxy(label: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    _, x_c, y_c, w, h = label.strip().split()
    x_c = float(x_c) * img_w
    y_c = float(y_c) * img_h
    w = float(w) * img_w
    h = float(h) * img_h
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return x1, y1, x2, y2


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        boxes.append(yolo_to_xyxy(line, img_w, img_h))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute F1-score at IoU=0.5")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--labels", required=True, help="Path to YOLO labels folder")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels folder not found: {labels_dir}")

    model = YOLO(args.weights)

    tp = 0
    fp = 0
    fn = 0

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        gt = load_gt_labels(labels_dir / f"{img_path.stem}.txt", img_w, img_h)

        preds = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        pred_boxes: List[Tuple[float, float, float, float]] = []
        if preds.boxes is not None:
            for b in preds.boxes:
                xyxy = b.xyxy.squeeze().tolist()
                if len(xyxy) == 4:
                    pred_boxes.append(tuple(map(float, xyxy)))

        matched_gt = set()
        for p in pred_boxes:
            best_iou = 0.0
            best_idx = None
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                val = iou(p, g)
                if val > best_iou:
                    best_iou = val
                    best_idx = i
            if best_iou >= args.iou and best_idx is not None:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += max(0, len(gt) - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1@IoU{args.iou:.2f}: {f1:.4f}")


if __name__ == "__main__":
    main()
