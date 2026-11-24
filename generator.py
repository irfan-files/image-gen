from __future__ import annotations

import gc
import json
import platform
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from hardware import detect_hardware, recommended_backend
from metadata import listing_text, write_json
from model_adapters.diffusers_adapter import DiffusersAdapter
from model_adapters.openvino_adapter import OpenVINOAdapter
from model_catalog import MODEL_OPTIONS, get_model_option
from prompt_builder import build_negative, build_prompt
from quality_control import analyze_image


class GenerationError(RuntimeError):
    pass


def load_themes(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in data["themes"]}


def deterministic_sku(theme_id: str, seed: int, width: int, height: int) -> str:
    import hashlib
    digest = hashlib.sha256(f"{theme_id}:{seed}:{width}:{height}".encode()).hexdigest()[:10].upper()
    return f"SAF-{theme_id[:10].upper()}-{digest}"


class Generator:
    def __init__(self, backend="auto", model_id=None, model_key=None, cache_dir="models",
                 openvino_model_dir=None, steps=28, guidance=None,
                 scheduler="dpmsolver", style_preset=""):
        self.info = detect_hardware()
        self.backend = recommended_backend(backend)
        if model_key is None:
            model_key = "sdxl" if platform.system() == "Darwin" else "sd15"
        if model_key not in MODEL_OPTIONS:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from {', '.join(MODEL_OPTIONS)}")
        self.model_key = model_key
        self.model_option = get_model_option(model_key)
        if self.backend == "openvino" and model_key != "sd15":
            raise ValueError("The OpenVINO adapter currently supports only the Stable Diffusion 1.5 preset.")
        self.cache_dir = cache_dir
        self.model_id = model_id or self.model_option.model_id
        self.openvino_model_dir = openvino_model_dir
        self.steps = steps
        self.guidance = guidance
        self.scheduler = scheduler
        self.style_preset = style_preset
        self.adapter = None

    def create_adapter(self):
        if self.backend == "openvino":
            return OpenVINOAdapter(self.model_id, self.cache_dir, self.openvino_model_dir)
        return DiffusersAdapter(
            self.model_id, self.backend, self.cache_dir, model_key=self.model_key
        )

    def ensure_adapter(self):
        if self.adapter is None:
            self.adapter = self.create_adapter()

    def unload(self):
        if self.adapter:
            self.adapter.unload()
        gc.collect()

    def generate_one(self, theme: dict, output_dir: Path, width=6000, height=3000,
                     seed=None, extra_prompt="", extra_negative="", aquarium_contrast="medium",
                     variant=0, skip_existing=True, ai_upscale=True) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        if seed is None:
            seed = random.SystemRandom().randrange(1, 2**32)
        sku = deterministic_sku(theme["id"], seed, width, height)
        sku_dir = output_dir / sku
        source = sku_dir / "source.png"
        metadata_path = sku_dir / "metadata.json"
        if skip_existing and source.exists() and metadata_path.exists():
            return {"status": "skipped", "sku": sku, "path": str(source), "seed": seed}

        extra = (self.style_preset + " " + extra_prompt).strip()
        prompt = build_prompt(theme, extra, seed, aquarium_contrast, variant)
        negative = build_negative(theme, extra_negative)

        resolutions = [(width, height)]
        if width >= 1536:
            if self.backend == "xpu":
                base_width, base_height = self.model_option.xpu_resolution
                smaller_width = max(128, (base_width * 5 // 6 // 8) * 8)
                smaller_height = max(128, (base_height * 5 // 6 // 8) * 8)
                smallest_width = max(128, (base_width * 2 // 3 // 8) * 8)
                smallest_height = max(128, (base_height * 2 // 3 // 8) * 8)
                resolutions = [
                    (base_width, base_height),
                    (smaller_width, smaller_height),
                    (smallest_width, smallest_height),
                ]
            else:
                resolutions = [(1536, 768), (1344, 672), (1216, 608), (1024, 512)]
        elif width >= 768:
            resolutions = [(width, height), (768, 384), (640, 320), (512, 256)]

        started = time.monotonic()
        last_error = None
        image = None
        actual_resolution = None
        for rw, rh in resolutions:
            try:
                self.ensure_adapter()
                image = self.adapter.generate(prompt, negative, rw, rh, seed,
                                              self.steps, self.guidance, self.scheduler)
                actual_resolution = (rw, rh)
                break
            except Exception as exc:
                last_error = exc
                if "out of memory" not in str(exc).lower() and "oom" not in str(exc).lower():
                    break
                self.unload()
        if image is None:
            raise GenerationError(f"Generation failed after retries: {last_error}")

        sku_dir.mkdir(parents=True, exist_ok=True)
        tmp_source = source.with_suffix(".png.tmp")
        image.save(tmp_source, format="PNG", dpi=(150, 150))
        tmp_source.replace(source)

        print_path = sku_dir / f"print_{width}x{height}.png"
        style_family = theme.get("style_family", "standard")
        pixel_palette_colors = int(theme.get("pixel_palette_colors", 32))
        pixel_art_era = theme.get("pixel_art_era")
        pixel_grid_height = theme.get("pixel_grid_height")
        if style_family == "pixel_art":
            from upscale import pixel_art_resize

            print_image, pixel_cell_size = pixel_art_resize(
                image, width, height, colors=pixel_palette_colors,
                grid_height=int(pixel_grid_height) if pixel_grid_height is not None else None,
            )
            tmp_print = print_path.with_suffix(".png.tmp")
            print_image.save(tmp_print, format="PNG", dpi=(150, 150), optimize=True)
            tmp_print.replace(print_path)
            upscale_method = "Not used for pixel art; palette quantization + nearest-neighbor"
            print_processing_method = (
                f"Pixel-art palette quantization ({pixel_palette_colors} colors) + "
                f"nearest-neighbor scaling (~{pixel_cell_size}px cells)"
            )
        elif ai_upscale:
            try:
                from upscale import upscale_one
                upscale_result = upscale_one(
                    source, print_path, width, height,
                    model="realesrnet_x4plus", tile=256, overlap=32,
                    use_realesrgan=True,
                )
                upscale_method = f"{upscale_result['method']} + exact-size Lanczos framing"
            except Exception as exc:
                raise GenerationError(
                    f"AI upscaling failed for {sku}; no low-resolution print master was saved: {exc}"
                ) from exc
            print_processing_method = upscale_method
        else:
            from upscale import cover_resize

            resized = cover_resize(image, width, height)
            tmp_print = print_path.with_suffix(".png.tmp")
            resized.save(tmp_print, format="PNG", dpi=(150, 150))
            tmp_print.replace(print_path)
            upscale_method = "Not used (AI upscaling disabled)"
            print_processing_method = "Lanczos framing and resize"

        preview = sku_dir / "preview.jpg"
        preview_tmp = preview.with_suffix(".jpg.tmp")
        with Image.open(print_path) as print_image:
            preview_image = print_image.convert("RGB")
        preview_resampling = (
            Image.Resampling.NEAREST
            if style_family == "pixel_art"
            else Image.Resampling.LANCZOS
        )
        preview_image.thumbnail((1280, 1280), preview_resampling)
        preview_image.save(preview_tmp, format="JPEG", quality=92, dpi=(150, 150), optimize=True)
        preview_tmp.replace(preview)

        qc = analyze_image(print_path, width, height)
        etsy_title, etsy_tags, etsy_desc = listing_text(theme, sku, "etsy")
        shopee_title, _, shopee_desc = listing_text(theme, sku, "shopee")
        data = {
            "SKU": sku, "Filename": print_path.name, "Theme": theme["name"],
            "Category": theme["category"], "Marketplace": ",".join(theme["marketplace"]),
            "Style": theme["art_style"], "Main color": theme["palette"], "Prompt": prompt,
            "NegativePrompt": negative, "Seed": seed, "Model ID": self.model_id,
            "Model key": self.model_key,
            "Model license": self.model_option.license_name,
            "Model license URL": self.model_option.license_url,
            "Commercial use note": self.model_option.commercial_note,
            "Width": width, "Height": height, "Print PPI": 150,
            "AI upscale method": upscale_method,
            "Print processing method": print_processing_method,
            "Style family": style_family,
            "Pixel art tier": pixel_art_era if style_family == "pixel_art" else None,
            "Pixel art grid height": pixel_grid_height if style_family == "pixel_art" else None,
            "Recommended model": theme.get("recommended_model"),
            "Pixel palette colors": pixel_palette_colors if style_family == "pixel_art" else None,
            "Quality status": qc["status"], "Etsy title": etsy_title,
            "Etsy tags": etsy_tags, "Etsy description": etsy_desc,
            "Shopee title": shopee_title, "Shopee description": shopee_desc,
            "Source path": str(source), "Master path": str(print_path), "Preview path": str(preview),
            "Quality": qc, "ActualResolution": actual_resolution,
            "Backend": self.backend, "Device": self.info.backend,
            "GeneratedAtUTC": datetime.now(timezone.utc).isoformat(), "DPI": 150,
        }
        write_json(metadata_path, data)
        elapsed = time.monotonic() - started
        data["ElapsedSeconds"] = round(elapsed, 3)
        data["status"] = "success"
        data["path"] = str(print_path)
        return data
