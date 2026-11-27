from __future__ import annotations

import hashlib
import random
from typing import Iterable

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
WALL_ART_OUTPUT = "flat 2D full-bleed standalone illustration, artwork only, fills the entire canvas"
UNDERWATER_OUTPUT = (
    "flat 2D full-bleed underwater wallpaper illustration, scene only, fills the entire canvas"
)
PIXEL_ART_STYLE_PROMPTS = {
    "8-bit": "8-bit-inspired pixel art, chunky square pixels, compact palette, hard edges",
    "16-bit": "16-bit-inspired pixel art, crisp square pixels, balanced palette, hard edges",
    "32-bit": "32-bit-inspired pixel art, fine square pixels, nuanced palette, hard edges",
}


def build_prompt(theme: dict, extra_prompt: str = "", seed: int | None = None,
                aquarium_contrast: str = "medium", variant: int = 0) -> str:
    palette = theme.get("palette", "balanced natural colors")
    aquarium = theme.get("category") == "aquarium"
    pixel_art = theme.get("style_family") == "pixel_art"
    rng = random.Random(seed if seed is not None else variant)
    palette_variant = PALETTE_VARIANTS[(variant + rng.randrange(0, len(PALETTE_VARIANTS))) % len(PALETTE_VARIANTS)]
    composition = theme.get("composition", "balanced panoramic composition")
    if aquarium:
        aquarium_line = (
            f"{composition}, "
            f"{aquarium_contrast}-contrast underwater scene, open central swim lane, "
            "aquatic details framing the side edges"
        )
        perspective_line = "perspective: straight-on panoramic underwater view"
    else:
        aquarium_line = composition
        perspective_line = f"perspective: {theme['perspective']}"

    output_type = UNDERWATER_OUTPUT if aquarium else WALL_ART_OUTPUT

    if pixel_art:
        pixel_style = PIXEL_ART_STYLE_PROMPTS.get(
            theme.get("pixel_art_era", "16-bit"),
            PIXEL_ART_STYLE_PROMPTS["16-bit"],
        )
        parts = [
            output_type,
            pixel_style,
            theme["primary_subject"],
            theme["environment"],
            theme["composition"],
            f"palette: {palette}",
            extra_prompt.strip(),
        ]
    else:
        parts = [
            output_type,
            f"{theme['art_style']} artwork",
            theme["primary_subject"],
            theme["environment"],
            theme["lighting"],
            f"palette: {palette}, {palette_variant}",
            aquarium_line,
            perspective_line,
            TEXTURE_VARIANTS[variant % len(TEXTURE_VARIANTS)],
            extra_prompt.strip(),
            QUALITY_SUFFIX,
        ]
    prompt = ", ".join(p for p in parts if p)
    return prompt


def build_negative(theme: dict, extra_negative: str = "") -> str:
    presentation_negative = (
        "product mockup, framed print, picture frame, poster mounted on wall, "
        "wall display, gallery wall, multiple framed artworks, duplicate artwork, "
        "diptych, collage, staged product photo, product catalog photo, "
        "external room around the artwork, "
        "person, human figure, figurine, statue, miniature diorama, 3D product render"
    )
    aquarium_negative = (
        "physical aquarium tank, fish tank, glass tank walls, aquarium glass, tank rim, "
        "aquarium stand, filter, pump, cables, fishbowl"
        if theme.get("category") == "aquarium"
        else ""
    )
    quality_negative = (
        "photorealistic, vector art, smooth gradients, antialiased edges, soft focus, "
        "blur, text, logo, watermark"
        if theme.get("style_family") == "pixel_art"
        else "blurry, low resolution, jpeg artifacts, oversharpened, bad anatomy, deformed"
    )
    return ", ".join(
        p for p in [
            theme.get("negative_prompt", ""),
            extra_negative,
            presentation_negative,
            aquarium_negative,
            quality_negative,
        ] if p
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
