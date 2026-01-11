from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw
from ultralytics import YOLO


def yolo_to_xyxy(label: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    parts = [p.replace("\\n", "") for p in label.strip().split()]
    _, x_c, y_c, w, h = parts
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
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    return [yolo_to_xyxy(line, img_w, img_h) for line in lines if line.strip()]


def draw_boxes(img: Image.Image, gt: List[Tuple[float, float, float, float]], preds: List[Tuple[float, float, float, float]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in gt:
        d.rectangle(b, outline="green", width=3)
    for b in preds:
        d.rectangle(b, outline="red", width=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export success/fail examples with boxes")
    parser.add_argument("--images", required=True, help="Path to images")
    parser.add_argument("--labels", required=True, help="Path to labels")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for match")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir = Path(args.out)
    success_dir = out_dir / "success"
    fail_dir = out_dir / "fail"
    success_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

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

        matched = False
        for p in pred_boxes:
            for g in gt:
                if iou(p, g) >= args.iou:
                    matched = True
                    break
            if matched:
                break

        out_img = draw_boxes(img, gt, pred_boxes)
        if matched:
            out_img.save(success_dir / img_path.name, quality=95)
        else:
            out_img.save(fail_dir / img_path.name, quality=95)

    print(f"Saved examples to: {out_dir}")


if __name__ == "__main__":
    main()
