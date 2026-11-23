from __future__ import annotations

from PIL import Image

from .base import BaseGenerator


class OpenVINOAdapter(BaseGenerator):
    """Adapter for an OpenVINO-exported Stable Diffusion 1.5 pipeline.

    It intentionally validates the dependency and exported model directory before
    claiming the backend is ready. A compatible export must contain OpenVINO .xml/.bin files.
    """
    def __init__(self, model_id: str, cache_dir: str, ov_dir: str | None = None):
        super().__init__(model_id, "openvino", cache_dir)
        self.ov_dir = ov_dir

    def load(self) -> None:
        try:
            import openvino
            from optimum.intel.openvino import OVStableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO/Optimum-Intel are not installed. Install requirements-windows.txt "
                "and export/download a verified OpenVINO SD1.5 model."
            ) from exc
        if not self.ov_dir:
            raise RuntimeError("OpenVINO model directory is required (set openvino_model_dir)")
        self.pipe = OVStableDiffusionPipeline.from_pretrained(
            self.ov_dir, cache_dir=self.cache_dir, device="GPU"
        )
        self.loaded = True

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                 seed: int, steps: int, guidance: float, scheduler: str) -> Image.Image:
        if not self.loaded:
            self.load()
        result = self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                           width=width, height=height, num_inference_steps=steps,
                           guidance_scale=guidance, generator=None)
        return result.images[0].convert("RGB")

    def unload(self) -> None:
        self.pipe = None
        super().unload()
