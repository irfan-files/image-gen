from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageStat


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def phash(image: Image.Image, hash_size: int = 16) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(gray, dtype=np.float32)
    from PIL import ImageFilter
    blurred = np.asarray(gray.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    mean = float(blurred.mean())
    bits = blurred > mean
    return "".join("1" if b else "0" for b in bits.ravel())


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def analyze_image(path: Path, expected_width: int | None = None,
                  expected_height: int | None = None) -> dict:
    issues: list[str] = []
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            mode = im.mode
            if im.format not in ("PNG", "JPEG", "JPG"):
                issues.append("unsupported_file_format")
            if expected_width and w != expected_width:
                issues.append("wrong_width")
            if expected_height and h != expected_height:
                issues.append("wrong_height")
            if mode not in ("RGB", "L", "RGBA"):
                issues.append("unexpected_color_mode")
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            mean = float(arr.mean())
            gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            contrast = float(gray.std())
            low_clip = float((arr < 0.01).mean())
            high_clip = float((arr > 0.99).mean())
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            small = rgb.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
            edge = np.asarray(small, dtype=np.float32)
            gx = np.diff(edge, axis=1)
            gy = np.diff(edge, axis=0)
            blur_proxy = float(np.sqrt((gx*gx).mean() + (gy*gy).mean()))
            hash_value = phash(rgb)
    except Exception as exc:
        return {"path": str(path), "valid": False, "issues": ["corrupt_or_unreadable", str(exc)],
                "checksum": None}

    if blur_proxy < 2.0: issues.append("possible_blur")
    if contrast < 0.08: issues.append("low_contrast")
    if low_clip > 0.08: issues.append("excessive_black_clipping")
    if high_clip > 0.08: issues.append("excessive_white_clipping")

    status = "NEEDS_REVIEW" if issues else "DRAFT"
    return {
        "path": str(path), "valid": not issues, "issues": issues, "status": status,
        "width": w, "height": h, "mode": mode, "brightness": mean,
        "contrast": contrast, "blur_proxy": blur_proxy,
        "low_clip_ratio": low_clip, "high_clip_ratio": high_clip,
        "phash": hash_value, "checksum": sha256(path), "file_size": path.stat().st_size,
    }


def find_duplicates(paths: list[Path], threshold: int = 5) -> list[tuple[Path, Path, int]]:
    records = [(p, phash(Image.open(p).convert("RGB"))) for p in paths]
    duplicates = []
    for i, (p1, h1) in enumerate(records):
        for p2, h2 in records[i+1:]:
            distance = hamming(h1, h2)
            if distance <= threshold:
                duplicates.append((p1, p2, distance))
    return duplicates


def contact_sheet(paths: list[Path], output: Path, columns: int = 4,
                  thumb_size: tuple[int, int] = (320, 180)) -> Path:
    from PIL import ImageDraw
    valid = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(thumb_size)
            valid.append((p, im.copy()))
        except Exception:
            pass
    if not valid:
        raise ValueError("No valid images for contact sheet")
    rows = (len(valid) + columns - 1) // columns
    sheet = Image.new("RGB", (columns*thumb_size[0], rows*(thumb_size[1]+35)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, im) in enumerate(valid):
        x = (i % columns)*thumb_size[0]; y = (i // columns)*(thumb_size[1]+35)
        sheet.paste(im, (x + (thumb_size[0]-im.width)//2, y))
        draw.text((x+8, y+thumb_size[1]+5), p.name[:32], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=90)
    return output
