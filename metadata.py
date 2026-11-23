from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

COLUMNS = [
    "SKU", "Filename", "Theme", "Category", "Marketplace", "Style", "Main color",
    "Prompt", "Seed", "Model ID", "Model license", "Width", "Height", "Print PPI",
    "AI upscale method", "Quality status", "Etsy title", "Etsy tags", "Etsy description",
    "Shopee title", "Shopee description", "Source path", "Master path", "Preview path"
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def listing_text(theme: dict, sku: str, marketplace: str) -> tuple[str, str, str]:
    name = theme["name"]
    if marketplace == "etsy":
        title = f"AI-Generated {name} Printable Wall Art | Digital Download"
        tags = ", ".join(theme["keywords"][:13])
        desc = (
            f"A human-curated, AI-assisted {name.lower()} digital artwork created with an original prompt. "
            "This listing is for a printable digital download, not a physical item. "
            "Due to file and color variation, verify print sizing and preview carefully."
        )
        return title, tags, desc
    title = f"Karya Seni Digital AI - {name}"
    desc = (
        f"Karya seni {name.lower()} yang dibuat dengan AI dan dikurasi manusia. "
        "Produk digital untuk dicetak sendiri; bukan karya fotografi asli atau barang fisik."
    )
    return title, "", desc


def write_export_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_export_excel(path: Path, rows: Iterable[dict]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/openpyxl required for Excel export") from exc
    pd.DataFrame(list(rows), columns=COLUMNS).to_excel(path, index=False)
