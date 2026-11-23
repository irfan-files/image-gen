from pathlib import Path
import json
from types import SimpleNamespace

import torch
from PIL import Image

import upscale
from generator import GenerationError, Generator, load_themes
from model_catalog import MODEL_OPTIONS


def test_rrdbnet_produces_four_times_resolution():
    model = upscale._build_rrdbnet(num_blocks=1).eval()
    with torch.inference_mode():
        result = model(torch.rand(1, 3, 8, 12))
    assert result.shape == (1, 3, 32, 48)


def test_realesrgan_upscale_writes_exact_size_and_reports_ai_method(tmp_path, monkeypatch):
    class FakeUpscaler(torch.nn.Module):
        def forward(self, image):
            return torch.nn.functional.interpolate(image, scale_factor=4, mode="nearest")

    monkeypatch.setattr(
        upscale, "load_realesrgan",
        lambda _model: (FakeUpscaler(), torch.device("cpu"), torch.float32),
    )
    source = tmp_path / "source.png"
    destination = tmp_path / "print.png"
    Image.new("RGB", (16, 8), (40, 100, 160)).save(source)

    result = upscale.upscale_one(
        source, destination, 64, 32,
        model="realesrnet_x4plus", tile=8, overlap=2, use_realesrgan=True,
    )

    assert result["method"] == "Real-ESRGAN x4"
    assert Image.open(destination).size == (64, 32)


def test_realesrgan_missing_weights_reports_expected_location(tmp_path, monkeypatch):
    monkeypatch.setattr(upscale, "__file__", str(tmp_path / "upscale.py"))

    try:
        upscale.load_realesrgan("realesrnet_x4plus")
    except RuntimeError as exc:
        assert "RealESRGAN_x4plus.pth" in str(exc)
        assert "models/realesrgan" in str(exc)
    else:
        raise AssertionError("Missing checkpoint must be reported")


def test_generation_does_not_silently_save_lanczos_when_ai_upscale_fails(tmp_path, monkeypatch):
    class FakeAdapter:
        def generate(self, *_args):
            return Image.new("RGB", (64, 32), (50, 80, 120))

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
    def fail_upscale(*_args, **_kwargs):
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr(upscale, "upscale_one", fail_upscale)
    theme_path = Path(__file__).resolve().parents[1] / "themes.yaml"
    theme = load_themes(theme_path)["vintage_winter_cabin"]

    try:
        generator.generate_one(theme, tmp_path, width=64, height=32, seed=10, ai_upscale=True)
    except GenerationError as exc:
        assert "AI upscaling failed" in str(exc)
        assert "checkpoint failed" in str(exc)
    else:
        raise AssertionError("AI upscaling failure must fail the generation job")

    sku = next(tmp_path.iterdir())
    assert (sku / "source.png").exists()
    assert not (sku / "print_64x32.png").exists()


def test_preview_is_created_from_upscaled_print_master(tmp_path, monkeypatch):
    class FakeAdapter:
        def generate(self, *_args):
            return Image.new("RGB", (64, 32), (40, 80, 120))

    def fake_upscale(_source, destination, width, height, **_kwargs):
        Image.new("RGB", (width, height), (200, 180, 160)).save(destination)
        return {"method": "Real-ESRGAN x4"}

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
    monkeypatch.setattr(upscale, "upscale_one", fake_upscale)
    theme_path = Path(__file__).resolve().parents[1] / "themes.yaml"
    theme = load_themes(theme_path)["vintage_winter_cabin"]

    result = generator.generate_one(
        theme, tmp_path, width=64, height=32, seed=11, ai_upscale=True
    )

    sku = tmp_path / result["SKU"]
    metadata = json.loads((sku / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["AI upscale method"].startswith("Real-ESRGAN x4")
    with Image.open(sku / "preview.jpg") as preview:
        assert preview.getpixel((preview.width // 2, preview.height // 2))[0] > 170
