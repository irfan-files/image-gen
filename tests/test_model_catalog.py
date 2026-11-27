from types import SimpleNamespace

from PIL import Image

from model_adapters.diffusers_adapter import (
    DiffusersAdapter,
    _configure_xpu_fp64_fallback,
    _encode_sd15_prompt,
    _pad_prompt_embeddings,
    _required_sequence_length,
    _split_clip_prompt,
)
from model_catalog import MODEL_OPTIONS


class FakeClipTokenizer:
    model_max_length = 6
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def _tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [sum(token.encode("utf-8")) + 3 for token in tokens]


def test_long_clip_prompt_splits_at_model_limit_without_losing_words():
    tokenizer = FakeClipTokenizer()
    prompt = "one two three four five six seven eight nine"

    chunks = _split_clip_prompt(prompt, [tokenizer])

    assert [word for chunk in chunks for word in chunk.split()] == prompt.split()
    assert all(len(tokenizer._tokenize(chunk)) <= 4 for chunk in chunks)


def test_long_unbroken_clip_word_splits_without_dropping_characters():
    class SubwordTokenizer:
        model_max_length = 6

        def __init__(self, subword_size):
            self.subword_size = subword_size

        def _tokenize(self, text):
            return [
                word[index:index + self.subword_size]
                for word in text.split()
                for index in range(0, len(word), self.subword_size)
            ]

        def convert_tokens_to_ids(self, tokens):
            return list(range(len(tokens)))

    prompt = "abcdefghijklmnopqrstu"
    tokenizers = [SubwordTokenizer(2), SubwordTokenizer(3)]

    chunks = _split_clip_prompt(prompt, tokenizers)

    assert "".join(chunks) == prompt
    assert all(
        len(tokenizer._tokenize(chunk)) <= 4
        for tokenizer in tokenizers
        for chunk in chunks
    )


def test_clip_prompt_count_ignores_persistent_backend_padding_and_truncation():
    class BackendTokenizer:
        truncation = True
        padding = True

        def no_truncation(self):
            self.truncation = False

        def no_padding(self):
            self.padding = False

        def encode(self, text, add_special_tokens=False):
            ids = [1] * len(text.split())
            if self.truncation or self.padding:
                ids = [1] * 77
            return SimpleNamespace(ids=ids)

    class StatefulTokenizer:
        model_max_length = 77

        def __init__(self):
            self.backend_tokenizer = BackendTokenizer()

    tokenizer = StatefulTokenizer()

    chunks = _split_clip_prompt("snowy village Christmas night", [tokenizer])

    assert chunks == ["snowy village Christmas night"]
    assert tokenizer.backend_tokenizer.truncation is False
    assert tokenizer.backend_tokenizer.padding is False


def test_prompt_limits_fail_explicitly_instead_of_truncating():
    tokenizer = FakeClipTokenizer()

    try:
        _split_clip_prompt(
            " ".join(f"word{i}" for i in range(13)),
            [tokenizer],
            max_chunks=2,
        )
    except ValueError as exc:
        assert "2-chunk" in str(exc)
    else:
        raise AssertionError("Overlong CLIP prompts must not be silently truncated")

    try:
        _required_sequence_length(tokenizer, ("one two three four five six",), 1, 6)
    except ValueError as exc:
        assert "6-token" in str(exc)
    else:
        raise AssertionError("Overlong sequence prompts must fail before inference")


def test_sd15_long_prompt_encodes_every_chunk_and_pads_negative_embeddings():
    import torch

    class FakeTextEncoder:
        config = SimpleNamespace(max_position_embeddings=6, use_attention_mask=True)

        def __call__(self, input_ids, attention_mask=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            return (input_ids.to(torch.float32).unsqueeze(-1),)

    pipe = SimpleNamespace(tokenizer=FakeClipTokenizer(), text_encoder=FakeTextEncoder())
    positive = _encode_sd15_prompt(
        pipe, "one two three four five six seven eight nine", torch.device("cpu")
    )
    negative = _encode_sd15_prompt(pipe, "no frame", torch.device("cpu"))

    assert positive.shape == (1, 15, 1)
    assert negative.shape == (1, 4, 1)
    negative = _pad_prompt_embeddings(negative, positive.shape[1])
    assert negative.shape == positive.shape
    assert pipe.text_encoder.input_ids.shape == (1, 6)
    assert pipe.text_encoder.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0]]


def test_sd15_adapter_passes_full_long_prompt_embeddings_to_diffusers():
    import torch

    class FakeTextEncoder:
        config = SimpleNamespace(max_position_embeddings=6, use_attention_mask=False)

        def __call__(self, input_ids):
            return (input_ids.to(torch.float32).unsqueeze(-1),)

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        text_encoder = FakeTextEncoder()
        _execution_device = torch.device("cpu")

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sd15"].model_id, "cpu", "models", model_key="sd15"
    )
    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate(
        "one two three four five six seven eight nine",
        "avoid frame",
        64,
        64,
        18,
        4,
        None,
        "default",
    )

    assert "prompt" not in adapter.pipe.kwargs
    assert "negative_prompt" not in adapter.pipe.kwargs
    assert adapter.pipe.kwargs["prompt_embeds"].shape == (
        1, adapter.pipe.kwargs["negative_prompt_embeds"].shape[1], 1
    )
    assert adapter.pipe.kwargs["prompt_embeds"].shape[1] > 6


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


def test_sdxl_loads_vae_in_float32_without_changing_other_component_dtype(monkeypatch):
    import diffusers
    import torch

    class FakeVae:
        dtype = torch.float32

        def enable_slicing(self):
            pass

        def decode(self, latents, **_kwargs):
            self.decoded_dtype = latents.dtype

    class FakePipeline:
        vae = FakeVae()

        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            cls.kwargs = kwargs
            return cls()

        def enable_sequential_cpu_offload(self, **_kwargs):
            pass

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(diffusers, "StableDiffusionXLPipeline", FakePipeline)
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sdxl"].model_id, "xpu", "models", model_key="sdxl"
    )

    adapter.load()

    assert FakePipeline.kwargs["dtype"] == {
        "default": torch.float16,
        "vae": torch.float32,
    }
    adapter.pipe.vae.decode(torch.ones((1,), dtype=torch.float16))
    assert adapter.pipe.vae.decoded_dtype == torch.float32


def test_flux_adapter_forwards_flux_sequence_limit_and_model_default_guidance():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-dev"].model_id, "cpu", "models", model_key="flux-dev"
    )
    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    prompt = "one two three four five six seven eight nine"
    negative = "avoid frame"
    image = adapter.generate(prompt, negative, 64, 64, 17, 4, None, "default")
    assert image.size == (64, 64)
    assert adapter.pipe.kwargs["max_sequence_length"] == 512
    assert adapter.pipe.kwargs["guidance_scale"] == 3.5
    assert adapter.pipe.kwargs["negative_prompt"] == negative
    assert adapter.pipe.kwargs["prompt_2"] == prompt
    assert adapter.pipe.kwargs["negative_prompt_2"] == negative
    assert len(adapter.pipe.tokenizer._tokenize(adapter.pipe.kwargs["prompt"])) <= 4
    assert adapter.pipe.kwargs["true_cfg_scale"] == 2.0


def test_flux_without_negative_prompt_does_not_enable_true_cfg():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-dev"].model_id, "cpu", "models", model_key="flux-dev"
    )

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    prompt = "one two three four five six seven eight nine"
    adapter.generate(prompt, "", 64, 64, 17, 4, None, "default")
    assert "true_cfg_scale" not in adapter.pipe.kwargs
    assert adapter.pipe.kwargs["prompt_2"] == prompt
    assert len(adapter.pipe.tokenizer._tokenize(adapter.pipe.kwargs["prompt"])) <= 4


def test_sd35_adapter_uses_its_pipeline_prompt_limit():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sd35-medium"].model_id, "cpu", "models", model_key="sd35-medium"
    )

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()
        tokenizer_3 = FakeClipTokenizer()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    prompt = "one two three four five six seven eight nine"
    adapter.generate(prompt, "negative", 64, 64, 18, 4, None, "default")
    assert adapter.pipe.kwargs["max_sequence_length"] == 256
    assert adapter.pipe.kwargs["guidance_scale"] == 4.5
    assert adapter.pipe.kwargs["prompt_3"] == prompt
    assert len(adapter.pipe.tokenizer._tokenize(adapter.pipe.kwargs["prompt"])) <= 4
    assert adapter.pipe.kwargs["negative_prompt_3"] == "negative"


def test_sd35_adapter_sends_full_prompt_to_t5_and_short_anchor_to_clip():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sd35-medium"].model_id, "cpu", "models", model_key="sd35-medium"
    )

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()
        tokenizer_3 = FakeClipTokenizer()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    prompt = "one two three four five six seven eight nine"
    adapter.generate(prompt, "avoid frame", 64, 64, 18, 4, None, "default")

    assert adapter.pipe.kwargs["prompt_3"] == prompt
    assert len(adapter.pipe.tokenizer._tokenize(adapter.pipe.kwargs["prompt"])) <= 4
    assert adapter.pipe.kwargs["negative_prompt_3"] == "avoid frame"


def test_sdxl_long_prompt_uses_all_chunks_for_positive_and_negative():
    import torch

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()
        _execution_device = torch.device("cpu")

        def encode_prompt(self, **kwargs):
            self.encoded_chunks = kwargs
            count = len(kwargs["prompt"])
            positive = torch.ones((count, 6, 2))
            negative = torch.zeros((count, 6, 2))
            pooled = torch.ones((count, 3))
            negative_pooled = torch.zeros((count, 3))
            return positive, negative, pooled, negative_pooled

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter = DiffusersAdapter(
        MODEL_OPTIONS["sdxl"].model_id, "cpu", "models", model_key="sdxl"
    )
    adapter.pipe = Pipeline()
    adapter.loaded = True
    prompt = " ".join(f"subject{i}" for i in range(13))
    adapter.generate(prompt, "avoid frame", 64, 64, 19, 4, None, "default")

    assert len(adapter.pipe.encoded_chunks["prompt"]) > 1
    assert "prompt" not in adapter.pipe.kwargs
    assert adapter.pipe.kwargs["prompt_embeds"].shape[1] > 6
    assert adapter.pipe.kwargs["negative_prompt_embeds"].shape == (
        1, adapter.pipe.kwargs["prompt_embeds"].shape[1], 2
    )


def test_flux_schnell_uses_distilled_inference_defaults():
    adapter = DiffusersAdapter(
        MODEL_OPTIONS["flux-schnell"].model_id, "cpu", "models", model_key="flux-schnell"
    )

    class Pipeline:
        tokenizer = FakeClipTokenizer()
        tokenizer_2 = FakeClipTokenizer()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    adapter.pipe = Pipeline()
    adapter.loaded = True
    adapter.generate("short prompt", "ignored negative", 64, 64, 21, 4, 7.5, "default")
    assert adapter.pipe.kwargs["guidance_scale"] == 0.0
    assert adapter.pipe.kwargs["num_inference_steps"] == 4
    assert adapter.pipe.kwargs["max_sequence_length"] == 256
    assert adapter.pipe.kwargs["prompt_2"] == "short prompt"
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
        tokenizer = FakeClipTokenizer()

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
