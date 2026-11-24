import json
import logging
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

from generator import Generator, deterministic_sku, load_themes
from hardware import _TorchRedirectNoteFilter, _TORCH_REDIRECT_LOGGER, _TORCH_REDIRECT_NOTE
from model_catalog import MODEL_OPTIONS
from metadata import write_export_csv
from prompt_builder import build_negative, build_prompt
from quality_control import analyze_image, phash, hamming
from upscale import cover_resize

ROOT = Path(__file__).resolve().parents[1]


def test_themes_complete():
    themes = load_themes(ROOT / "themes.yaml")
    assert len(themes) >= 30
    required = {"id", "name", "category", "marketplace", "primary_subject", "environment",
                "art_style", "palette", "lighting", "perspective", "composition",
                "detail_level", "negative_prompt", "target_buyer", "keywords",
                "seasonality", "commercial_notes"}
    for theme in themes.values():
        assert required <= theme.keys()
    pixel_themes = [theme for theme in themes.values() if theme.get("style_family") == "pixel_art"]
    assert len(pixel_themes) >= 10
    assert all(theme.get("recommended_model") == "sd15" for theme in pixel_themes)
    assert any(theme["category"] == "new_year_wall_art" for theme in pixel_themes)
    assert any(theme["category"] == "aquarium" for theme in pixel_themes)


def test_deterministic_sku():
    a = deterministic_sku("vintage_winter_cabin", 123, 6000, 3000)
    b = deterministic_sku("vintage_winter_cabin", 123, 6000, 3000)
    assert a == b


def test_cover_resize_no_distortion_and_exact_size():
    image = Image.new("RGB", (800, 400), "red")
    output = cover_resize(image, 400, 400)
    assert output.size == (400, 400)


def test_quality_control(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (600, 300), (128, 128, 128)).save(path)
    result = analyze_image(path, 600, 300)
    assert result["width"] == 600 and result["height"] == 300
    assert result["checksum"]


def test_duplicate_hash():
    a = Image.new("RGB", (128, 128), "blue")
    b = a.copy()
    assert hamming(phash(a), phash(b)) == 0


def test_prompt_contains_constraints():
    themes = load_themes(ROOT / "themes.yaml")
    prompt = build_prompt(themes["vintage_winter_cabin"])
    assert "no text" in prompt.lower()
    assert "no watermark" in prompt.lower()


def test_pixel_art_prompt_uses_style_constraints_and_aquarium_open_center():
    themes = load_themes(ROOT / "themes.yaml")
    prompt = build_prompt(themes["pixel_nano_planted_aquarium"]).lower()
    negative = build_negative(themes["pixel_nano_planted_aquarium"]).lower()

    assert "pixel art" in prompt
    assert "square pixel clusters" in prompt
    assert "open swimming lane" in prompt
    assert "no antialiasing" in prompt
    assert "photorealism" in negative
    assert len(prompt.split()) <= 60


def test_csv_export(tmp_path):
    path = tmp_path / "catalog.csv"
    write_export_csv(path, [{"SKU": "TEST", "Filename": "test.png"}])
    assert path.exists()


def test_torch_redirect_filter_suppresses_only_expected_platform_note():
    note = logging.LogRecord(_TORCH_REDIRECT_LOGGER, logging.WARNING, "", 0, _TORCH_REDIRECT_NOTE, (), None)
    other_warning = logging.LogRecord(
        _TORCH_REDIRECT_LOGGER, logging.WARNING, "", 0, "another warning", (), None
    )
    warning_from_other_logger = logging.LogRecord(
        "torch.other", logging.WARNING, "", 0, _TORCH_REDIRECT_NOTE, (), None
    )
    filter_ = _TorchRedirectNoteFilter()

    assert not filter_.filter(note)
    assert filter_.filter(other_warning)
    assert filter_.filter(warning_from_other_logger)


def test_pixel_art_generation_uses_crisp_print_renderer_not_ai_upscaler(tmp_path, monkeypatch):
    class FakeAdapter:
        def generate(self, *_args):
            image = Image.new("RGB", (64, 32))
            for x in range(64):
                for y in range(32):
                    image.putpixel((x, y), (255, 0, 0) if x < 32 else (0, 0, 255))
            return image

    def fail_if_ai_upscaler_used(*_args, **_kwargs):
        raise AssertionError("Pixel art should not run Real-ESRGAN")

    monkeypatch.setattr("upscale.upscale_one", fail_if_ai_upscaler_used)
    themes = load_themes(ROOT / "themes.yaml")
    theme = themes["pixel_nano_planted_aquarium"]
    generator = Generator.__new__(Generator)
    generator.backend = "cpu"
    generator.model_key = "sd15"
    generator.model_option = MODEL_OPTIONS["sd15"]
    generator.model_id = MODEL_OPTIONS["sd15"].model_id
    generator.info = SimpleNamespace(backend="cpu")
    generator.adapter = FakeAdapter()
    generator.steps = 1
    generator.guidance = None
    generator.scheduler = "default"
    generator.style_preset = ""

    result = generator.generate_one(
        theme, tmp_path, width=64, height=32, seed=10, ai_upscale=True
    )

    sku = tmp_path / result["SKU"]
    metadata = json.loads((sku / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["Style family"] == "pixel_art"
    assert "nearest-neighbor" in metadata["Print processing method"]
    assert metadata["AI upscale method"].startswith("Not used for pixel art")
    with Image.open(sku / "print_64x32.png") as print_image:
        assert print_image.size == (64, 32)
        assert len(print_image.getcolors(maxcolors=257)) <= 32
        assert print_image.getpixel((0, 0)) == print_image.getpixel((7, 7))
