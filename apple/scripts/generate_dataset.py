from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SVG_EXTS = {".svg"}


def load_logo(path: Path) -> Image.Image | None:
    ext = path.suffix.lower()
    if ext in RASTER_EXTS:
        img = Image.open(path).convert("RGBA")
    elif ext in SVG_EXTS:
        try:
            import cairosvg  # optional
        except Exception:
            return None
        png_bytes = cairosvg.svg2png(url=str(path))
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    else:
        return None

    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    return img.crop(bbox)


def random_background(w: int, h: int) -> Image.Image:
    mode = random.choice(["solid", "gradient", "noise", "stripes"])
    if mode == "solid":
        color = tuple(random.randint(0, 255) for _ in range(3))
        return Image.new("RGB", (w, h), color)
    if mode == "gradient":
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        t = np.linspace(0, 1, h).reshape(h, 1, 1)
        grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        grad = np.repeat(grad, w, axis=1)
        return Image.fromarray(grad, mode="RGB")
    if mode == "noise":
        noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(noise, mode="RGB")
    base = Image.new("RGB", (w, h), tuple(random.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(base)
    for y in range(0, h, random.randint(20, 60)):
        stripe_color = tuple(random.randint(0, 255) for _ in range(3))
        d.rectangle([0, y, w, y + random.randint(8, 20)], fill=stripe_color)
    return base


def colorize_logo(logo: Image.Image) -> Image.Image:
    if random.random() < 0.7:
        color = tuple(random.randint(0, 255) for _ in range(3))
        colored = Image.new("RGB", logo.size, color)
        colored.putalpha(logo.split()[3])
        return colored
    return logo


def write_yolo_label(path: Path, box: Tuple[float, float, float, float]) -> None:
    x_c, y_c, w, h = box
    path.write_text(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from transparent logos")
    parser.add_argument("--src", required=True, help="Folder with transparent logos")
    parser.add_argument("--out", required=True, help="Output folder (data/...)")
    parser.add_argument("--num", type=int, default=1200, help="Number of images to generate")
    parser.add_argument("--split", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    labeled_images = out_dir / "labeled" / "images"
    labeled_labels = out_dir / "labeled" / "labels"
    train_dir = out_dir / "splits" / "train"
    val_dir = out_dir / "splits" / "val"

    for p in [
        labeled_images,
        labeled_labels,
        train_dir / "images",
        train_dir / "labels",
        val_dir / "images",
        val_dir / "labels",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # clear old files
    for p in labeled_images.glob("*"):
        p.unlink()
    for p in labeled_labels.glob("*"):
        p.unlink()
    for p in (train_dir / "images").glob("*"):
        p.unlink()
    for p in (train_dir / "labels").glob("*"):
        p.unlink()
    for p in (val_dir / "images").glob("*"):
        p.unlink()
    for p in (val_dir / "labels").glob("*"):
        p.unlink()

    logos: List[Image.Image] = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file():
            img = load_logo(p)
            if img is not None:
                logos.append(img)
    if not logos:
        raise SystemExit("No usable logos found in src folder")

    img_sizes = [(640, 640), (512, 512), (800, 600), (600, 800), (1024, 768)]

    for i in range(args.num):
        bg_w, bg_h = random.choice(img_sizes)
        bg = random_background(bg_w, bg_h)

        logo = colorize_logo(random.choice(logos))
        target = random.uniform(0.2, 0.6) * min(bg_w, bg_h)
        scale = target / max(logo.size)
        new_w = max(1, int(logo.width * scale))
        new_h = max(1, int(logo.height * scale))
        logo = logo.resize((new_w, new_h), Image.BICUBIC)

        angle = random.uniform(-25, 25)
        logo = logo.rotate(angle, expand=True, resample=Image.BICUBIC)

        if logo.width >= bg_w or logo.height >= bg_h:
            shrink = min(bg_w / (logo.width + 1), bg_h / (logo.height + 1)) * 0.9
            logo = logo.resize(
                (max(1, int(logo.width * shrink)), max(1, int(logo.height * shrink))),
                Image.BICUBIC,
            )

        x = random.randint(0, bg_w - logo.width)
        y = random.randint(0, bg_h - logo.height)
        bg.paste(logo, (x, y), logo)

        alpha = logo.split()[3]
        bbox = alpha.getbbox()
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        x_min += x
        x_max += x
        y_min += y
        y_max += y

        x_c = (x_min + x_max) / 2.0 / bg_w
        y_c = (y_min + y_max) / 2.0 / bg_h
        w = (x_max - x_min) / bg_w
        h = (y_max - y_min) / bg_h

        if random.random() < 0.25:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

        img_name = f"logo_{i:05d}.jpg"
        label_name = f"logo_{i:05d}.txt"
        bg.save(labeled_images / img_name, quality=95)
        write_yolo_label(labeled_labels / label_name, (x_c, y_c, w, h))

    images = sorted(labeled_images.glob("*"))
    random.shuffle(images)
    split_idx = int(args.split * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def copy_pair(img_path: Path, split_dir: Path) -> None:
        label_path = labeled_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            return
        (split_dir / "images").joinpath(img_path.name).write_bytes(img_path.read_bytes())
        (split_dir / "labels").joinpath(label_path.name).write_bytes(label_path.read_bytes())

    for img in train_images:
        copy_pair(img, train_dir)
    for img in val_images:
        copy_pair(img, val_dir)

    # write dataset yaml
    yaml_path = out_dir / "apple.yaml"
    yaml_path.write_text(
        "names:\\n- apple_logo\\npath: " + str((out_dir / "splits").resolve()) + "\\ntrain: train/images\\nval: val/images\\n",
        encoding="utf-8",
    )

    print(f"Generated: {len(images)} images")
    print(f"Train: {len(train_images)}  Val: {len(val_images)}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
