from __future__ import annotations

import gc

from PIL import Image

from hardware import configure_torch_logging
from model_catalog import MODEL_OPTIONS, get_model_option
from .base import BaseGenerator

MAX_CLIP_PROMPT_CHUNKS = 4
QWEN_IMAGE_21_MODEL_KEY = "qwen-image-2.1"
FLUX2_KLEIN_MODEL_KEYS = frozenset({
    "flux2-klein-9b", "flux2-klein-4b", "flux2-klein-base-4b"
})


def _configure_xpu_fp64_fallback(torch) -> None:
    import diffusers.utils.torch_utils as diffusers_torch_utils

    try:
        unsupported_devices = frozenset(
            diffusers_torch_utils._FP64_UNSUPPORTED_DEVICES
        ) | {"xpu"}
        dtype_unsupported_devices = diffusers_torch_utils._DTYPE_UNSUPPORTED_DEVICES
    except AttributeError as exc:
        raise RuntimeError(
            "This Diffusers version does not expose the dtype compatibility mapping "
            "needed to run FLUX on Intel XPU. Upgrade Diffusers and retry."
        ) from exc

    diffusers_torch_utils._FP64_UNSUPPORTED_DEVICES = unsupported_devices
    dtype_unsupported_devices[torch.float64] = unsupported_devices


def _token_ids(tokenizer, text: str) -> list[int]:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        backend_tokenizer.no_truncation()
        backend_tokenizer.no_padding()
        return backend_tokenizer.encode(text, add_special_tokens=False).ids
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        return encode(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
    tokenize = getattr(tokenizer, "_tokenize", None) or tokenizer.tokenize
    tokens = tokenize(text)
    return tokenizer.convert_tokens_to_ids(tokens)


def _clip_content_limit(tokenizer) -> int:
    model_max_length = getattr(tokenizer, "model_max_length", 77)
    if not isinstance(model_max_length, int) or model_max_length > 4096:
        model_max_length = 77
    return max(1, model_max_length - 2)


def _split_oversized_clip_word(word: str, tokenizers: list, limits: list[int]) -> list[str]:
    fragments = []
    remaining = word
    while remaining:
        low = 1
        high = len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            fragment = remaining[:middle]
            if all(
                len(_token_ids(tokenizer, fragment)) <= limit
                for tokenizer, limit in zip(tokenizers, limits)
            ):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError(
                "A prompt fragment cannot fit within a CLIP tokenizer's context limit."
            )
        fragments.append(remaining[:best])
        remaining = remaining[best:]
    return fragments


def _split_clip_prompt(
    text: str,
    tokenizers: list,
    max_chunks: int = MAX_CLIP_PROMPT_CHUNKS,
    first_only: bool = False,
) -> list[str]:
    if not tokenizers:
        return [text]

    limits = [_clip_content_limit(tokenizer) for tokenizer in tokenizers]
    def fits(candidate: str) -> bool:
        return all(
            len(_token_ids(tokenizer, candidate)) <= limit
            for tokenizer, limit in zip(tokenizers, limits)
        )

    chunks: list[str] = []
    current = ""

    for word in text.split():
        candidate = f"{current} {word}".strip()
        if fits(candidate):
            current = candidate
            continue

        if current:
            chunks.append(current)
            if first_only:
                return chunks
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"Prompt exceeds the supported {max_chunks}-chunk CLIP limit; "
                    "shorten the prompt or extra prompt text."
                )
            current = ""

        if fits(word):
            current = word
            continue

        for fragment in _split_oversized_clip_word(word, tokenizers, limits):
            candidate = current + fragment
            if fits(candidate):
                current = candidate
                continue
            chunks.append(current)
            if first_only:
                return chunks
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"Prompt exceeds the supported {max_chunks}-chunk CLIP limit; "
                    "shorten the prompt or extra prompt text."
                )
            current = fragment

    if current or not chunks:
        chunks.append(current)
    if len(chunks) > max_chunks:
        raise ValueError(
            f"Prompt exceeds the supported {max_chunks}-chunk CLIP limit; "
            "shorten the prompt or extra prompt text."
        )
    return chunks


def _encode_sd15_prompt(pipe, text: str, device):
    import torch

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    max_length = int(text_encoder.config.max_position_embeddings)
    payload_size = max_length - 2
    token_ids = _token_ids(tokenizer, text)
    chunks = [
        token_ids[index:index + payload_size]
        for index in range(0, len(token_ids), payload_size)
    ] or [[]]
    if len(chunks) > MAX_CLIP_PROMPT_CHUNKS:
        raise ValueError(
            f"Prompt exceeds the supported {MAX_CLIP_PROMPT_CHUNKS}-chunk CLIP limit; "
            "shorten the prompt or extra prompt text."
        )

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    input_batch = []
    attention_masks = []
    valid_lengths = []
    for chunk in chunks:
        input_ids = [bos_id, *chunk, eos_id]
        valid_length = len(input_ids)
        input_ids.extend([pad_id] * (max_length - valid_length))
        input_batch.append(input_ids)
        valid_lengths.append(valid_length)
        attention_masks.append(
            [1] * valid_length + [0] * (max_length - valid_length)
        )
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    encoder_kwargs = {}
    if getattr(text_encoder.config, "use_attention_mask", False):
        encoder_kwargs["attention_mask"] = torch.tensor(
            attention_masks, dtype=torch.long, device=device
        )
    with torch.inference_mode():
        encoded = text_encoder(input_tensor, **encoder_kwargs)[0]
    return torch.cat(
        [
            encoded[index:index + 1, :valid_length, :]
            for index, valid_length in enumerate(valid_lengths)
        ],
        dim=1,
    )


def _pad_prompt_embeddings(prompt_embeds, target_length: int):
    if prompt_embeds.shape[1] >= target_length:
        return prompt_embeds
    padding = prompt_embeds[:, -1:, :].expand(
        -1, target_length - prompt_embeds.shape[1], -1
    )
    import torch

    return torch.cat((prompt_embeds, padding), dim=1)


def _required_sequence_length(tokenizer, prompts: tuple[str, ...], minimum: int, maximum: int) -> int:
    token_count = max((len(_token_ids(tokenizer, text)) for text in prompts), default=0) + 1
    if token_count > maximum:
        raise ValueError(
            f"Prompt exceeds this model's {maximum}-token text-encoder limit; "
            "shorten the prompt or extra prompt text."
        )
    return max(minimum, token_count)


def _cast_sdxl_vae_decode_input(vae, dtype) -> None:
    original_decode = vae.decode

    def decode(latents, *args, **kwargs):
        if latents.dtype != dtype:
            latents = latents.to(dtype=dtype)
        return original_decode(latents, *args, **kwargs)

    vae.decode = decode


class DiffusersAdapter(BaseGenerator):
    def __init__(
        self,
        model_id: str,
        backend: str,
        cache_dir: str,
        model_key: str = "sd15",
        dtype: str = "auto",
    ):
        super().__init__(model_id, backend, cache_dir)
        self.model_key = model_key
        self.model_option = get_model_option(model_key)
        self.dtype = dtype
        self.pipe = None

    def load(self) -> None:
        configure_torch_logging()
        try:
            import torch
            import diffusers
        except ImportError as exc:
            raise RuntimeError("Install torch and diffusers in the selected Python environment.") from exc

        if self.backend == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "The selected PyTorch build cannot access Intel XPU. Install the tested XPU build "
                "and verify the Intel graphics driver, or select CPU."
            )

        if self.backend == "xpu" and self.model_key in {
            "flux-dev", "flux-schnell", *FLUX2_KLEIN_MODEL_KEYS
        }:
            _configure_xpu_fp64_fallback(torch)

        pipeline_class_name = self.model_option.pipeline_class
        pipeline_class = getattr(diffusers, pipeline_class_name, None)
        if pipeline_class is None:
            raise RuntimeError(
                f"Installed Diffusers {diffusers.__version__} does not expose {pipeline_class_name}. "
                "Upgrade Diffusers in the same environment that launches the app."
            )

        if self.backend == "mps":
            dtype = (
                torch.float16
                if self.model_key in {
                    "flux-dev", "flux-schnell", "sd35-medium", "z-image-turbo",
                    QWEN_IMAGE_21_MODEL_KEY,
                    *FLUX2_KLEIN_MODEL_KEYS,
                }
                else torch.float32
            )
        elif self.backend in ("cuda", "xpu"):
            if self.model_key in FLUX2_KLEIN_MODEL_KEYS | {QWEN_IMAGE_21_MODEL_KEY}:
                dtype = (
                    torch.bfloat16
                    if self.backend == "xpu" or torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                dtype = torch.float16
        elif self.model_key in FLUX2_KLEIN_MODEL_KEYS | {QWEN_IMAGE_21_MODEL_KEY}:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        load_dtype = (
            {"default": dtype, "vae": torch.float32}
            if self.model_key == "sdxl"
            else dtype
        )

        try:
            if self.model_key == "flux-3-klein":
                components_manager = None
                if self.backend != "cpu":
                    from diffusers.modular_pipelines import ComponentsManager

                    components_manager = ComponentsManager()
                    self.pipe = pipeline_class.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                        components_manager=components_manager,
                    )
                else:
                    self.pipe = pipeline_class.from_pretrained(
                        self.model_id,
                        cache_dir=self.cache_dir,
                    )
                if self.backend == "mps":
                    model_dtype = torch.float16
                elif self.backend == "cuda" and not torch.cuda.is_bf16_supported():
                    model_dtype = torch.float16
                else:
                    model_dtype = torch.bfloat16
                self.pipe.load_components(
                    cache_dir=self.cache_dir,
                    dtype={"default": model_dtype, "vae": torch.float32},
                )
                missing_components = [
                    name
                    for name, spec in self.pipe._component_specs.items()
                    if spec.default_creation_method == "from_pretrained"
                    and spec.pretrained_model_name_or_path is not None
                    and not hasattr(self.pipe, name)
                ]
                if missing_components:
                    missing = ", ".join(sorted(missing_components))
                    raise RuntimeError(
                        f"Could not load required FLUX 3 Image Klein components: {missing}. "
                        "Check model access approval, Hugging Face authentication, and network access."
                    )
                if components_manager is not None:
                    components_manager.enable_auto_cpu_offload(
                        device=torch.device(self.backend)
                    )
            else:
                self.pipe = pipeline_class.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    dtype=load_dtype,
                    use_safetensors=True,
                )
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                raise RuntimeError(
                    f"{self.model_id} requires model access approval. Accept its terms on "
                    f"{self.model_option.license_url} and authenticate with `hf auth login`."
                ) from exc
            raise RuntimeError(f"Could not load {self.model_id}: {exc}") from exc

        if self.model_key == "sdxl":
            _cast_sdxl_vae_decode_input(self.pipe.vae, torch.float32)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        if self.model_key != "flux-3-klein":
            if self.backend in {"xpu", "mps"}:
                if self.model_key in {
                    "flux-dev", "sd35-medium", "flux-schnell", "z-image-turbo",
                    QWEN_IMAGE_21_MODEL_KEY,
                    *FLUX2_KLEIN_MODEL_KEYS,
                } or (
                    self.backend == "xpu" and self.model_key == "sdxl"
                ):
                    try:
                        self.pipe.enable_sequential_cpu_offload(device=torch.device(self.backend))
                    except Exception as exc:
                        self.pipe = None
                        raise RuntimeError(
                            f"{self.model_key} needs layer-by-layer CPU/RAM offload on {self.backend}, "
                            f"but Diffusers could not enable it: {exc}"
                        ) from exc
                elif self.backend == "xpu":
                    try:
                        self.pipe.enable_model_cpu_offload(device=torch.device("xpu"))
                    except TypeError:
                        self.pipe.enable_model_cpu_offload()
                    except Exception as exc:
                        self.pipe = None
                        raise RuntimeError(f"Could not enable XPU model offload: {exc}") from exc
                else:
                    try:
                        self.pipe.to("mps")
                    except Exception as exc:
                        self.pipe = None
                        raise RuntimeError(f"Could not move Diffusers pipeline to MPS: {exc}") from exc
            elif self.backend == "cuda":
                if self.model_key in FLUX2_KLEIN_MODEL_KEYS | {QWEN_IMAGE_21_MODEL_KEY}:
                    try:
                        self.pipe.enable_model_cpu_offload(device=torch.device("cuda"))
                    except Exception as exc:
                        self.pipe = None
                        raise RuntimeError(
                            f"{self.model_key} needs model CPU offload on CUDA, "
                            f"but Diffusers could not enable it: {exc}"
                        ) from exc
                else:
                    try:
                        self.pipe.to(self.backend)
                    except Exception as exc:
                        self.pipe = None
                        raise RuntimeError(
                            f"Could not move Diffusers pipeline to {self.backend}: {exc}"
                        ) from exc

        self.loaded = True

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float | None,
        scheduler: str,
    ) -> Image.Image:
        if not self.loaded:
            self.load()
        import torch

        if self.model_key == "flux-schnell" and not 1 <= steps <= 4:
            raise ValueError("FLUX.1-schnell supports 1-4 inference steps; choose a value in that range.")

        if scheduler.lower() == "dpmsolver" and self.model_key in {"sd15", "sdxl"}:
            from diffusers import DPMSolverMultistepScheduler

            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        model_guidance = (
            0.0
            if self.model_key in {"flux-schnell", "z-image-turbo", "flux-3-klein"}
            else guidance
        )
        if model_guidance is None:
            model_guidance = {
                "sd15": 7.5,
                "sdxl": 7.5,
                "flux-dev": 3.5,
                "sd35-medium": 4.5,
                "flux-schnell": 0.0,
                "z-image-turbo": 0.0,
                "flux-3-klein": 0.0,
                "flux2-klein-9b": 1.0,
                "flux2-klein-4b": 1.0,
                "flux2-klein-base-4b": 4.0,
                QWEN_IMAGE_21_MODEL_KEY: 1.0,
            }[self.model_key]

        # CPU generators work with sequential offload and avoid accelerator generator mismatches.
        generator_device = (
            "cpu"
            if self.model_key in {
                "flux-dev", "sd35-medium", "flux-schnell", "z-image-turbo", "flux-3-klein",
                QWEN_IMAGE_21_MODEL_KEY,
                *FLUX2_KLEIN_MODEL_KEYS,
            }
            and self.backend in {"xpu", "mps"}
            else self.backend
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        do_classifier_free_guidance = model_guidance > 1.0
        call_args = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": model_guidance,
            "generator": generator,
        }
        if self.model_key not in {
            "flux-schnell", "z-image-turbo", "flux-3-klein", *FLUX2_KLEIN_MODEL_KEYS
        }:
            call_args["negative_prompt"] = negative_prompt
        if self.model_key == "flux-dev":
            call_args["max_sequence_length"] = 512
            if negative_prompt:
                call_args["true_cfg_scale"] = 2.0
        elif self.model_key == "flux-schnell":
            call_args["max_sequence_length"] = 256
        elif self.model_key == "sd35-medium":
            call_args["max_sequence_length"] = 256
        elif self.model_key == "z-image-turbo":
            call_args["max_sequence_length"] = 512
        elif self.model_key == "flux-3-klein":
            call_args.pop("guidance_scale")
            call_args["output"] = "images"
        elif self.model_key == QWEN_IMAGE_21_MODEL_KEY:
            call_args.pop("guidance_scale")
            call_args["true_cfg_scale"] = model_guidance
            if model_guidance <= 1.0:
                call_args.pop("negative_prompt", None)

        if self.model_key == "sd15":
            tokenizer = self.pipe.tokenizer
            text_encoder = self.pipe.text_encoder
            max_tokens = int(text_encoder.config.max_position_embeddings) - 2
            prompt_is_long = (
                len(_token_ids(tokenizer, prompt)) > max_tokens
                or (
                    do_classifier_free_guidance
                    and len(_token_ids(tokenizer, negative_prompt)) > max_tokens
                )
            )
            if prompt_is_long:
                execution_device = getattr(
                    self.pipe, "_execution_device", torch.device(self.backend)
                )
                prompt_embeds = _encode_sd15_prompt(
                    self.pipe, prompt, execution_device
                )
                call_args.pop("prompt")
                call_args["prompt_embeds"] = prompt_embeds
                if do_classifier_free_guidance:
                    negative_embeds = _encode_sd15_prompt(
                        self.pipe, negative_prompt, execution_device
                    )
                    target_length = max(
                        prompt_embeds.shape[1], negative_embeds.shape[1]
                    )
                    call_args["prompt_embeds"] = _pad_prompt_embeddings(
                        prompt_embeds, target_length
                    )
                    call_args["negative_prompt_embeds"] = _pad_prompt_embeddings(
                        negative_embeds, target_length
                    )
                call_args.pop("negative_prompt", None)
        elif self.model_key == "sdxl":
            clip_tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
            max_tokens = min(
                _clip_content_limit(tokenizer) for tokenizer in clip_tokenizers
            )
            prompt_is_long = (
                len(_token_ids(clip_tokenizers[0], prompt)) > max_tokens
                or len(_token_ids(clip_tokenizers[1], prompt)) > max_tokens
                or (
                    do_classifier_free_guidance
                    and (
                        len(_token_ids(clip_tokenizers[0], negative_prompt)) > max_tokens
                        or len(_token_ids(clip_tokenizers[1], negative_prompt)) > max_tokens
                    )
                )
            )
            if prompt_is_long:
                prompt_chunks = _split_clip_prompt(prompt, clip_tokenizers)
                negative_chunks = (
                    _split_clip_prompt(negative_prompt, clip_tokenizers)
                    if do_classifier_free_guidance
                    else []
                )
                positive_chunk_count = len(prompt_chunks)
                negative_chunk_count = len(negative_chunks)
                chunk_count = max(len(prompt_chunks), len(negative_chunks))
                prompt_chunks.extend([""] * (chunk_count - len(prompt_chunks)))
                negative_chunks.extend([""] * (chunk_count - len(negative_chunks)))
                execution_device = getattr(
                    self.pipe, "_execution_device", torch.device(self.backend)
                )
                encoded = self.pipe.encode_prompt(
                    prompt=prompt_chunks,
                    prompt_2=prompt_chunks,
                    device=execution_device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_chunks if do_classifier_free_guidance else None,
                    negative_prompt_2=negative_chunks if do_classifier_free_guidance else None,
                )
                positive, negative, pooled, negative_pooled = encoded
                call_args.pop("prompt")
                call_args.pop("negative_prompt", None)
                call_args["prompt_embeds"] = positive.reshape(
                    1, -1, positive.shape[-1]
                )
                call_args["pooled_prompt_embeds"] = pooled[
                    :positive_chunk_count
                ].mean(dim=0, keepdim=True)
                if do_classifier_free_guidance:
                    call_args["negative_prompt_embeds"] = negative.reshape(
                        1, -1, negative.shape[-1]
                    )
                    call_args["negative_pooled_prompt_embeds"] = negative_pooled[
                        :negative_chunk_count
                    ].mean(dim=0, keepdim=True)
        elif self.model_key in {"flux-dev", "flux-schnell"}:
            clip_prompt = _split_clip_prompt(
                prompt, [self.pipe.tokenizer], first_only=True
            )[0]
            call_args["prompt"] = clip_prompt
            call_args["prompt_2"] = prompt
            sequence_limit = 512 if self.model_key == "flux-dev" else 256
            call_args["max_sequence_length"] = _required_sequence_length(
                self.pipe.tokenizer_2,
                (prompt, negative_prompt),
                call_args["max_sequence_length"],
                sequence_limit,
            )
            if self.model_key == "flux-dev" and do_classifier_free_guidance and negative_prompt:
                clip_negative = _split_clip_prompt(
                    negative_prompt, [self.pipe.tokenizer], first_only=True
                )[0]
                call_args["negative_prompt"] = clip_negative
                call_args["negative_prompt_2"] = negative_prompt
        elif self.model_key == "sd35-medium":
            clip_tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
            clip_prompt = _split_clip_prompt(
                prompt, clip_tokenizers, first_only=True
            )[0]
            call_args["prompt"] = clip_prompt
            call_args["prompt_2"] = clip_prompt
            call_args["prompt_3"] = prompt
            sequence_length = _required_sequence_length(
                self.pipe.tokenizer_3,
                (prompt, negative_prompt),
                call_args["max_sequence_length"],
                512,
            )
            call_args["max_sequence_length"] = sequence_length
            if do_classifier_free_guidance:
                clip_negative = _split_clip_prompt(
                    negative_prompt, clip_tokenizers, first_only=True
                )[0]
                call_args["negative_prompt"] = clip_negative
                call_args["negative_prompt_2"] = clip_negative
                call_args["negative_prompt_3"] = negative_prompt
        elif self.model_key == "z-image-turbo":
            call_args["max_sequence_length"] = _required_sequence_length(
                self.pipe.tokenizer,
                (prompt, negative_prompt),
                call_args["max_sequence_length"],
                512,
            )

        try:
            result = self.pipe(**call_args)
        except Exception as exc:
            raise RuntimeError(
                f"{self.model_key} inference failed on {self.backend} at {width}x{height}: {exc}"
            ) from exc

        gc.collect()
        if self.backend in ("mps", "cuda", "xpu"):
            getattr(torch, self.backend).empty_cache()
        if self.model_key == "flux-3-klein":
            return result[0].convert("RGB")
        return result.images[0].convert("RGB")

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        super().unload()
