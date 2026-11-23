import logging
from pathlib import Path

import yaml
from PIL import Image

from generator import deterministic_sku, load_themes
from hardware import _TorchRedirectNoteFilter, _TORCH_REDIRECT_LOGGER, _TORCH_REDIRECT_NOTE
from prompt_builder import build_prompt
from quality_control import analyze_image, phash, hamming
from metadata import write_export_csv
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
