from __future__ import annotations

import hashlib
import random
from typing import Iterable

COMPOSITION_VARIANTS = [
    "harmonious panoramic composition",
    "subject balanced at the left third with generous clean space",
    "subject balanced at the right third with calm negative space",
    "layered depth from foreground through distant background",
]

PALETTE_VARIANTS = [
    "sophisticated restrained palette",
    "muted tonal palette with natural color balance",
    "high-contrast focal palette with calm surrounding areas",
]

TEXTURE_VARIANTS = [
    "finely detailed natural textures",
    "smooth fine-art surface with believable material detail",
    "subtle brush texture and dimensional surface detail",
]

QUALITY_SUFFIX = "visually coherent, strong depth, no text, no watermark, no logo"
PIXEL_ART_SUFFIX = (
    "crisp 16-bit pixel art, square pixel clusters, hard stepped edges, "
    "limited palette, no antialiasing"
)


def build_prompt(theme: dict, extra_prompt: str = "", seed: int | None = None,
                aquarium_contrast: str = "medium", variant: int = 0) -> str:
    palette = theme.get("palette", "balanced natural colors")
    aquarium = theme.get("category") == "aquarium"
    pixel_art = theme.get("style_family") == "pixel_art"
    rng = random.Random(seed if seed is not None else variant)
    composition_variant = COMPOSITION_VARIANTS[(variant + rng.randrange(0, len(COMPOSITION_VARIANTS))) % len(COMPOSITION_VARIANTS)]
    palette_variant = PALETTE_VARIANTS[(variant + rng.randrange(0, len(PALETTE_VARIANTS))) % len(PALETTE_VARIANTS)]
    aquarium_line = ""
    if aquarium:
        aquarium_line = (
            f"{theme.get('composition', 'clean panoramic composition')}, "
            f"{aquarium_contrast}-contrast aquarium background, ample open swimming area in the center, "
            "details concentrated toward side edges, easy to read through aquarium glass"
        )
    else:
        aquarium_line = f"{theme.get('composition', composition_variant)}, {composition_variant}"

    if pixel_art:
        parts = [
            PIXEL_ART_SUFFIX,
            theme["primary_subject"],
            theme["environment"],
            theme["composition"],
            f"palette: {palette}",
            extra_prompt.strip(),
        ]
    else:
        parts = [
            f"{theme['art_style']} artwork",
            theme["primary_subject"],
            theme["environment"],
            theme["lighting"],
            f"palette: {palette}, {palette_variant}",
            aquarium_line,
            f"perspective: {theme['perspective']}",
            TEXTURE_VARIANTS[variant % len(TEXTURE_VARIANTS)],
            extra_prompt.strip(),
            QUALITY_SUFFIX,
        ]
    prompt = ", ".join(p for p in parts if p)
    return prompt


def build_negative(theme: dict, extra_negative: str = "") -> str:
    quality_negative = (
        "photorealistic, vector art, smooth gradients, antialiased edges, soft focus, "
        "blur, text, logo, watermark"
        if theme.get("style_family") == "pixel_art"
        else "blurry, low resolution, jpeg artifacts, oversharpened, bad anatomy, deformed"
    )
    return ", ".join(
        p for p in [theme.get("negative_prompt", ""), extra_negative, quality_negative] if p
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
