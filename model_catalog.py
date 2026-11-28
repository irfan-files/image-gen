from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    model_id: str
    pipeline_class: str
    license_name: str
    license_url: str
    commercial_note: str
    xpu_resolution: tuple[int, int]
    minimum_ram_gb: int
    recommended_steps: int


MODEL_OPTIONS = {
    "sd15": ModelOption(
        key="sd15",
        label="Stable Diffusion 1.5 (Arc 4 GB + pixel art)",
        model_id="runwayml/stable-diffusion-v1-5",
        pipeline_class="StableDiffusionPipeline",
        license_name="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/runwayml/stable-diffusion-v1-5",
        commercial_note="Check the model card and license restrictions before selling outputs.",
        xpu_resolution=(768, 384),
        minimum_ram_gb=8,
        recommended_steps=28,
    ),
    "sdxl": ModelOption(
        key="sdxl",
        label="Stable Diffusion XL 1.0",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_class="StableDiffusionXLPipeline",
        license_name="CreativeML Open RAIL++-M",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        commercial_note="Review the model card and license restrictions for commercial use.",
        xpu_resolution=(640, 320),
        minimum_ram_gb=16,
        recommended_steps=28,
    ),
    "flux-dev": ModelOption(
        key="flux-dev",
        label="FLUX.1-dev (non-commercial license)",
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_class="FluxPipeline",
        license_name="FLUX.1-dev Non-Commercial License",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev",
        commercial_note=(
            "Non-commercial license: do not use for Etsy/Shopee sales unless you have "
            "a separate commercial license from Black Forest Labs."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=28,
    ),
    "sd35-medium": ModelOption(
        key="sd35-medium",
        label="Stable Diffusion 3.5 Medium",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        pipeline_class="StableDiffusion3Pipeline",
        license_name="Stability AI Community License",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
        commercial_note=(
            "Commercial use is permitted under the Community License for individuals/"
            "organizations below USD 1M annual revenue; above that requires an enterprise license."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=24,
        recommended_steps=28,
    ),
    "flux-schnell": ModelOption(
        key="flux-schnell",
        label="FLUX.1-schnell (Apache 2.0, 1-4 steps)",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class="FluxPipeline",
        license_name="Apache 2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        commercial_note=(
            "The model is released under Apache 2.0; comply with the license and applicable "
            "laws when using or selling generated work."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=4,
    ),
    "z-image-turbo": ModelOption(
        key="z-image-turbo",
        label="Z-Image-Turbo (Apache 2.0, fast)",
        model_id="Tongyi-MAI/Z-Image-Turbo",
        pipeline_class="ZImagePipeline",
        license_name="Apache 2.0",
        license_url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        commercial_note=(
            "The model is released under Apache 2.0. Its model card describes a 6B-parameter "
            "model and recommends consumer GPUs with about 16 GB VRAM; 4 GB Arc requires "
            "CPU/RAM offload and is likely to be very slow."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=9,
    ),
    "flux-3-klein": ModelOption(
        key="flux-3-klein",
        label="FLUX 3 Image Klein (gated, 16 steps)",
        model_id="black-forest-labs/flux-3-image-klein",
        pipeline_class="ModularPipeline",
        license_name="FLUX Kommunity License v.1.0",
        license_url="https://huggingface.co/black-forest-labs/flux-3-image-klein",
        commercial_note=(
            "The gated FLUX Kommunity License permits non-commercial use and commercial "
            "output generation by qualifying users below US$5M gross annualized revenue, "
            "subject to the full license and acceptable-use terms."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=64,
        recommended_steps=16,
    ),
    "flux2-klein-9b": ModelOption(
        key="flux2-klein-9b",
        label="FLUX.2 Klein 9B (non-commercial, 4 steps)",
        model_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline_class="Flux2KleinPipeline",
        license_name="FLUX Non-Commercial License",
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
        commercial_note=(
            "Non-commercial license: do not use for Etsy/Shopee sales. Hugging Face access "
            "approval is required, and the license requires applicable safety filters or manual review."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=64,
        recommended_steps=4,
    ),
    "flux2-klein-4b": ModelOption(
        key="flux2-klein-4b",
        label="FLUX.2 Klein 4B (Apache 2.0, 4 steps)",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline_class="Flux2KleinPipeline",
        license_name="Apache 2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        commercial_note=(
            "Apache 2.0 permits commercial use. Follow its terms and review generated images; "
            "the model card encourages appropriate safety filters and manual review."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=4,
    ),
    "flux2-klein-base-4b": ModelOption(
        key="flux2-klein-base-4b",
        label="FLUX.2 Klein Base 4B (Apache 2.0, 50 steps)",
        model_id="black-forest-labs/FLUX.2-klein-base-4B",
        pipeline_class="Flux2KleinPipeline",
        license_name="Apache 2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B",
        commercial_note=(
            "Apache 2.0 permits commercial use. Follow its terms and review generated images; "
            "the model card encourages appropriate safety filters and manual review."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=50,
    ),
    "ideogram-4-fp8": ModelOption(
        key="ideogram-4-fp8",
        label="Ideogram 4 FP8 (non-commercial, 48 steps)",
        model_id="ideogram-ai/ideogram-4-fp8",
        pipeline_class="Ideogram4Pipeline (official ideogram4 package)",
        license_name="Ideogram Non-Commercial Model Agreement",
        license_url="https://huggingface.co/ideogram-ai/ideogram-4-fp8",
        commercial_note=(
            "Non-commercial use only. Do not use generated images for Etsy/Shopee sales "
            "or other commercial purposes. The gated model requires Hugging Face access approval."
        ),
        xpu_resolution=(512, 256),
        minimum_ram_gb=32,
        recommended_steps=48,
    ),
    "qwen-image-2.1": ModelOption(
        key="qwen-image-2.1",
        label="Qwen Image 2.1 (research license, 40 steps)",
        model_id="Qwen/Qwen-Image-2.1",
        pipeline_class="QwenImage21Pipeline",
        license_name="Qwen Research License Agreement",
        license_url="https://huggingface.co/Qwen/Qwen-Image-2.1",
        commercial_note=(
            "The standard license permits non-commercial use only. Commercial use requires "
            "a separate license from Qwen; proceed with Etsy/Shopee generation only if you "
            "hold that license."
        ),
        xpu_resolution=(768, 384),
        minimum_ram_gb=64,
        recommended_steps=40,
    ),
}

MODEL_LABEL_TO_KEY = {option.label: key for key, option in MODEL_OPTIONS.items()}


def get_model_option(model_key: str) -> ModelOption:
    try:
        return MODEL_OPTIONS[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model '{model_key}'. Choose one of: {', '.join(MODEL_OPTIONS)}"
        ) from exc
