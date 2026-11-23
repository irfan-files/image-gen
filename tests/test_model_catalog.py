from types import SimpleNamespace

from PIL import Image

from model_adapters.diffusers_adapter import DiffusersAdapter, _configure_xpu_fp64_fallback
from model_catalog import MODEL_OPTIONS


def test_flux_xpu_uses_float32_for_unsupported_fp64_embeddings(monkeypatch):
    import torch
    import diffusers.utils.torch_utils as diffusers_torch_utils

    monkeypatch.setattr(
        diffusers_torch_utils,
        "_FP64_UNSUPPORTED_DEVICES",
        diffusers_torch_utils._FP64_UNSUPPORTED_DEVICES,
    )
    monkeypatch.setitem(
        diffusers_torch_utils._DTYPE_UNSUPPORTED_DEVICES,
        torch.float64,
        diffusers_torch_utils._DTYPE_UNSUPPORTED_DEVICES[torch.float64],
    )

    _configure_xpu_fp64_fallback(torch)

    assert diffusers_torch_utils.maybe_adjust_dtype_for_device(
        torch.float64, torch.device("xpu")
    ) == torch.float32


def test_optional_model_presets_use_correct_pipelines_and_licenses():
    assert MODEL_OPTIONS["flux-dev"].model_id == "black-forest-labs/FLUX.1-dev"
    assert MODEL_OPTIONS["flux-dev"].pipeline_class == "FluxPipeline"
    assert "Non-Commercial" in MODEL_OPTIONS["flux-dev"].license_name
    assert MODEL_OPTIONS["sd35-medium"].model_id == "stabilityai/stable-diffusion-3.5-medium"
    assert MODEL_OPTIONS["sd35-medium"].pipeline_class == "StableDiffusion3Pipeline"
    assert MODEL_OPTIONS["flux-schnell"].model_id == "black-forest-labs/FLUX.1-schnell"
    assert MODEL_OPTIONS["flux-schnell"].pipeline_class == "FluxPipeline"
    assert MODEL_OPTIONS["flux-schnell"].license_name == "Apache 2.0"
    assert MODEL_OPTIONS["flux-schnell"].recommended_steps == 4
    assert MODEL_OPTIONS["z-image-turbo"].model_id == "Tongyi-MAI/Z-Image-Turbo"
    assert MODEL_OPTIONS["z-image-turbo"].pipeline_class == "ZImagePipeline"
    assert MODEL_OPTIONS["z-image-turbo"].license_name == "Apache 2.0"
    assert MODEL_OPTIONS["z-image-turbo"].recommended_steps == 9


def test_diffusers_load_uses_current_dtype_argument(monkeypatch):
    import diffusers
    import torch

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            cls.kwargs = kwargs
            return cls()

    monkeypatch.setattr(diffusers, "StableDiffusionPipeline", FakePipeline)
    adapter = DiffusersAdapter(MODEL_OPTIONS["sd15"].model_id, "cpu", "models")

    adapter.load()

    assert FakePipeline.kwargs["dtype"] == torch.float32
    assert "torch_dtype" not in FakePipeline.kwargs


def test_flux_adapter_forwards_flux_sequence_limit_and_model_default_guidance():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-dev"].model_id, "cpu", "models", model_key="flux-dev"
    )
    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    image = adapter.generate("short prompt", "negative", 64, 64, 17, 4, None, "default")
    assert image.size == (64, 64)
    assert adapter.pipe.kwargs["max_sequence_length"] == 512
    assert adapter.pipe.kwargs["guidance_scale"] == 3.5
    assert adapter.pipe.kwargs["negative_prompt"] == "negative"
    assert adapter.pipe.kwargs["true_cfg_scale"] == 2.0


def test_flux_without_negative_prompt_does_not_enable_true_cfg():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-dev"].model_id, "cpu", "models", model_key="flux-dev"
    )

    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate("short prompt", "", 64, 64, 17, 4, None, "default")
    assert "true_cfg_scale" not in adapter.pipe.kwargs


def test_sd35_adapter_uses_its_pipeline_prompt_limit():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sd35-medium"].model_id, "cpu", "models", model_key="sd35-medium"
    )

    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate("short prompt", "negative", 64, 64, 18, 4, None, "default")
    assert adapter.pipe.kwargs["max_sequence_length"] == 256
    assert adapter.pipe.kwargs["guidance_scale"] == 4.5


def test_flux_schnell_uses_distilled_inference_defaults():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-schnell"].model_id, "cpu", "models", model_key="flux-schnell"
    )

    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate("short prompt", "ignored negative", 64, 64, 21, 4, 7.5, "default")
    assert adapter.pipe.kwargs["guidance_scale"] == 0.0
    assert adapter.pipe.kwargs["num_inference_steps"] == 4
    assert adapter.pipe.kwargs["max_sequence_length"] == 256
    assert "negative_prompt" not in adapter.pipe.kwargs


def test_flux_schnell_rejects_more_than_four_steps():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-schnell"].model_id, "cpu", "models", model_key="flux-schnell"
    )
    adapter.pipe = object()
    adapter.loaded = True

    try:
        adapter.generate("short prompt", "", 64, 64, 21, 5, None, "default")
    except ValueError as exc:
        assert "1-4" in str(exc)
    else:
        raise AssertionError("FLUX.1-schnell must reject inference steps above four")


def test_z_image_turbo_uses_supported_call_parameters():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["z-image-turbo"].model_id, "cpu", "models", model_key="z-image-turbo"
    )

    class Pipeline:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate("short prompt", "ignored negative", 64, 64, 22, 9, 7.5, "default")
    assert adapter.pipe.kwargs["guidance_scale"] == 0.0
    assert adapter.pipe.kwargs["num_inference_steps"] == 9
    assert adapter.pipe.kwargs["max_sequence_length"] == 512
    assert "negative_prompt" not in adapter.pipe.kwargs
