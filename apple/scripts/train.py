from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--data", required=True, help="Path to data yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="runs", help="Project folder")
    parser.add_argument("--name", default="apple_logo", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
