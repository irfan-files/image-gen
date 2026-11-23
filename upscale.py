from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

from PIL import Image


def load_realesrgan(model_name: str = "realesr-animevideov2"):
    """Load an optional RealESRGANer implementation if installed.

    Supports the real-esrgan package when the model .pth files are separately
    present. No model is downloaded or claimed usable without dependency checks.
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model_dir = Path(__file__).resolve().parent / "models" / "realesrgan"
        if model_name == "realesrnet_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            model_path = model_dir / "RealESRGAN_x4plus.pth"
            netscale = 4
        elif model_name == "realesrgan_x4plus_anime_6B":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            model_path = model_dir / "RealESRGAN_x4plus_anime_6B.pth"
            netscale = 4
        else:
            raise RuntimeError("Choose realesrnet_x4plus or realesrgan_x4plus_anime_6B")
        if not model_path.exists():
            raise RuntimeError(f"Missing official Real-ESRGAN weight: {model_path}")
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
        return RealESRGANer(scale=netscale, model_path=str(model_path), model=model,
                            tile=0, tile_pad=10, pre_pad=10,
                            half=(device.type != "cpu"), device=device)
    except ImportError as exc:
        raise RuntimeError("Install 'realesrgan basicsr' and obtain the matching official model file") from exc


def cover_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width*ratio), round(image.height*ratio)), Image.Resampling.LANCZOS)
    left = (resized.width-width)//2
    top = (resized.height-height)//2
    return resized.crop((left, top, left+width, top+height)).convert("RGB")


def upscale_one(source: Path, destination: Path, width: int, height: int,
                model: str, tile: int = 512, overlap: int = 32,
                use_realesrgan: bool = False) -> dict:
    source_img = Image.open(source).convert("RGB")
    if use_realesrgan:
        upsampler = load_realesrgan(model)
        upsampler.tile = tile
        upsampler.tile_pad = overlap
        import numpy as np
        output, _ = upsampler.enhance(np.asarray(source_img), outscale=4)
        gc.collect()
        enlarged = Image.fromarray(output, mode="RGB")
    else:
        # Explicit alternative: bicubic/Lanczos quality-preserving pipeline when
        # Real-ESRGAN cannot be installed or run, never presented as AI upscaling.
        enlarged = source_img.resize((source_img.width*2, source_img.height*2), Image.Resampling.LANCZOS)

    final = cover_resize(enlarged, width, height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    final.save(tmp, format="PNG", dpi=(150, 150), optimize=True)
    tmp.replace(destination)
    return {"source": str(source), "destination": str(destination), "width": width,
            "height": height, "method": "real_esrgan" if use_realesrgan else "lanczos_fallback",
            "file_size": destination.stat().st_size}


def main():
    p = argparse.ArgumentParser(description="Seasonal Art Factory upscaling")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--width", type=int, default=6000)
    p.add_argument("--height", type=int, default=3000)
    p.add_argument("--model", choices=["realesrnet_x4plus", "realesrgan_x4plus_anime_6B"], default="realesrnet_x4plus")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--use-realesrgan", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output or (str(input_path) + "_upscaled"))
    files = [input_path] if input_path.is_file() else list(input_path.rglob("*.png"))
    success = failed = 0
    for source in files:
        try:
            destination = output_dir / (source.parent.name + "_print.png") if input_path.is_dir() else output_dir / source.name
            result = upscale_one(source, destination, args.width, args.height,
                                 args.model, args.tile, args.overlap, args.use_realesrgan)
            success += 1
            print("OK", destination)
        except Exception as exc:
            failed += 1
            print("FAILED", source, exc, file=sys.stderr)
    print(json_summary(success, failed, output_dir))


def json_summary(success, failed, output):
    return f"Success: {success}  Failed: {failed}  Output: {output}"


if __name__ == "__main__":
    main()
