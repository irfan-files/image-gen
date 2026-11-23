from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _build_rrdbnet(num_blocks: int):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(96, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 32, 3, 1, 1)
            self.conv4 = nn.Conv2d(160, 32, 3, 1, 1)
            self.conv5 = nn.Conv2d(192, 64, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value):
            first = self.activation(self.conv1(value))
            second = self.activation(self.conv2(torch.cat((value, first), dim=1)))
            third = self.activation(self.conv3(torch.cat((value, first, second), dim=1)))
            fourth = self.activation(
                self.conv4(torch.cat((value, first, second, third), dim=1))
            )
            fifth = self.conv5(torch.cat((value, first, second, third, fourth), dim=1))
            return value + fifth * 0.2

    class ResidualInResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, value):
            return value + self.rdb3(self.rdb2(self.rdb1(value))) * 0.2

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
            self.body = nn.Sequential(
                *(ResidualInResidualDenseBlock() for _ in range(num_blocks))
            )
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, value):
            features = self.conv_first(value)
            features = features + self.conv_body(self.body(features))
            features = self.activation(
                self.conv_up1(functional.interpolate(features, scale_factor=2, mode="nearest"))
            )
            features = self.activation(
                self.conv_up2(functional.interpolate(features, scale_factor=2, mode="nearest"))
            )
            return self.conv_last(self.activation(self.conv_hr(features)))

    return RRDBNet()


def load_realesrgan(model_name: str = "realesrnet_x4plus"):
    """Load official Real-ESRGAN weights into a local RRDBNet implementation."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to enable Real-ESRGAN upscaling.") from exc

    model_config = {
        "realesrnet_x4plus": ("RealESRGAN_x4plus.pth", 23),
        "realesrgan_x4plus_anime_6B": ("RealESRGAN_x4plus_anime_6B.pth", 6),
    }
    if model_name not in model_config:
        raise ValueError(f"Unsupported Real-ESRGAN model: {model_name}")
    filename, block_count = model_config[model_name]
    model_path = Path(__file__).resolve().parent / "models" / "realesrgan" / filename
    if not model_path.exists():
        raise RuntimeError(
            f"Missing Real-ESRGAN weights: {model_path}. Download the official "
            f"{filename} file into models/realesrgan."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model = _build_rrdbnet(block_count)
    model.load_state_dict(state_dict, strict=True)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type in {"xpu", "cuda"} else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    return model, device, dtype


def _run_realesrgan(image: Image.Image, model_name: str, tile: int, overlap: int) -> Image.Image:
    import torch

    model, device, dtype = load_realesrgan(model_name)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.to(device=device, dtype=dtype)
    height, width = rgb.shape[:2]
    output = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
    tile = max(1, tile)
    overlap = max(0, overlap)

    with torch.inference_mode():
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                patch_top = max(0, top - overlap)
                patch_left = max(0, left - overlap)
                patch_bottom = min(height, bottom + overlap)
                patch_right = min(width, right + overlap)
                prediction = model(
                    input_tensor[:, :, patch_top:patch_bottom, patch_left:patch_right]
                )[0].float().clamp_(0, 1)
                prediction = (
                    prediction.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                )
                crop_top = (top - patch_top) * 4
                crop_left = (left - patch_left) * 4
                crop_bottom = crop_top + (bottom - top) * 4
                crop_right = crop_left + (right - left) * 4
                output[top * 4:bottom * 4, left * 4:right * 4] = prediction[
                    crop_top:crop_bottom, crop_left:crop_right
                ]

    del input_tensor, model
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return Image.fromarray(output, mode="RGB")


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
        enlarged = _run_realesrgan(source_img, model, tile, overlap)
    else:
        enlarged = source_img.resize((source_img.width*2, source_img.height*2), Image.Resampling.LANCZOS)

    final = cover_resize(enlarged, width, height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    final.save(tmp, format="PNG", dpi=(150, 150), optimize=True)
    tmp.replace(destination)
    return {"source": str(source), "destination": str(destination), "width": width,
            "height": height, "method": "Real-ESRGAN x4" if use_realesrgan else "Lanczos",
            "file_size": destination.stat().st_size}


def main():
    p = argparse.ArgumentParser(description="Seasonal Art Factory upscaling")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--width", type=int, default=6000)
    p.add_argument("--height", type=int, default=3000)
    p.add_argument("--model", choices=["realesrnet_x4plus", "realesrgan_x4plus_anime_6B"], default="realesrnet_x4plus")
    p.add_argument("--tile", type=int, default=256)
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
