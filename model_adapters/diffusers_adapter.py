from __future__ import annotations

import gc

from PIL import Image

from model_catalog import MODEL_OPTIONS, get_model_option
from .base import BaseGenerator


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

        if self.backend == "xpu" and self.model_key in {"flux-dev", "flux-schnell"}:
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
                    "flux-dev", "flux-schnell", "sd35-medium", "z-image-turbo"
                }
                else torch.float32
            )
        elif self.backend in ("cuda", "xpu"):
            dtype = torch.float16
        else:
            dtype = torch.float32

        try:
            self.pipe = pipeline_class.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=dtype,
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

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        if self.backend in {"xpu", "mps"}:
            if self.model_key in {"flux-dev", "sd35-medium", "flux-schnell", "z-image-turbo"} or (
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
            try:
                self.pipe.to(self.backend)
            except Exception as exc:
                self.pipe = None
                raise RuntimeError(f"Could not move Diffusers pipeline to {self.backend}: {exc}") from exc

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

        model_guidance = 0.0 if self.model_key in {"flux-schnell", "z-image-turbo"} else guidance
        if model_guidance is None:
            model_guidance = {
                "sd15": 7.5,
                "sdxl": 7.5,
                "flux-dev": 3.5,
                "sd35-medium": 4.5,
                "flux-schnell": 0.0,
                "z-image-turbo": 0.0,
            }[self.model_key]

        # CPU generators work with sequential offload and avoid accelerator generator mismatches.
        generator_device = (
            "cpu"
            if self.model_key in {"flux-dev", "sd35-medium", "flux-schnell", "z-image-turbo"}
            and self.backend in {"xpu", "mps"}
            else self.backend
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        call_args = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": model_guidance,
            "generator": generator,
        }
        if self.model_key not in {"flux-schnell", "z-image-turbo"}:
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

        try:
            result = self.pipe(**call_args)
        except Exception as exc:
            raise RuntimeError(
                f"{self.model_key} inference failed on {self.backend} at {width}x{height}: {exc}"
            ) from exc

        gc.collect()
        if self.backend in ("mps", "cuda", "xpu"):
            getattr(torch, self.backend).empty_cache()
        return result.images[0].convert("RGB")

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        super().unload()
