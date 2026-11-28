from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import platform
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:
    if exc.name == "yaml":
        raise SystemExit(
            "PyYAML is missing from this Python environment. "
            "Launch start_gui.bat or install it with: python -m pip install PyYAML"
        ) from None
    raise

from hardware import detect_hardware
from generator import Generator, load_themes
from metadata import write_export_csv, write_export_excel
from model_catalog import MODEL_OPTIONS, get_model_option
from quality_control import contact_sheet


PRINT_MODES = {
    "master": (6000, 3000), "large_print": (12000, 6000),
    "etsy_2_3": (4000, 6000), "etsy_3_4": (4500, 6000),
    "etsy_4_5": (4800, 6000), "iso_a4": (4961, 7016),
    "frame_tv": (3840, 2160), "aquarium_2_1": (6000, 3000),
    "aquarium_3_1": (9000, 3000),
}


def parse_args():
    p = argparse.ArgumentParser(description="Seasonal Art Factory - local batch artwork generator")
    p.add_argument("--theme")
    p.add_argument(
        "--category",
        choices=["christmas_wall_art", "new_year_wall_art", "winter_landscape", "aquarium"],
    )
    p.add_argument("--count", type=int, default=1,
                   help="Total number of images; ignored for --count-per-theme and --theme-counts")
    p.add_argument("--count-per-theme", type=int,
                   help="Generate this many unique images for every selected theme")
    p.add_argument("--theme-counts", default="",
                   help="Dynamic per-theme counts, for example: frozen_crystal_cave:10,vintage_winter_cabin:25")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda", "xpu", "openvino"])
    p.add_argument("--model", default=None, choices=list(MODEL_OPTIONS),
                   help="Model preset from the model catalog")
    p.add_argument("--model-id")
    p.add_argument("--input", default="themes.yaml")
    p.add_argument("--output", default="outputs")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--print-mode", choices=list(PRINT_MODES), default="master")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, default=None,
                   help="Inference steps; defaults to a model-specific recommendation")
    p.add_argument("--guidance", type=float, default=None,
                   help="Guidance scale; default is model-specific")
    p.add_argument("--scheduler", default="dpmsolver")
    p.add_argument("--style-preset", default="")
    p.add_argument("--extra-prompt", default="")
    p.add_argument("--extra-negative", default="")
    p.add_argument("--aquarium-contrast", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--list-themes", action="store_true")
    p.add_argument("--interactive", action="store_true",
                   help="Choose themes and enter a unique image count for each theme")
    p.add_argument("--queue", default="",
                   help="YAML batch queue file with items: theme_id/count")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--export-excel", action="store_true")
    p.add_argument("--ai-upscale", action="store_true", default=True,
                   help="Upscale the print master with Real-ESRGAN x4 (requires official weights)")
    p.add_argument("--no-ai-upscale", action="store_false", dest="ai_upscale")
    p.add_argument("--openvino-model")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    themes = load_themes(input_path)

    if args.list_themes:
        for theme_id, theme in themes.items():
            print(f"{theme_id} | {theme['name']} | {theme['category']} | count={len(theme.get('prompt_variants', [])) or 'dynamic'}")
        return

    selected = []
    if args.theme:
        if args.theme not in themes:
            raise SystemExit(f"Unknown theme '{args.theme}'. Available: {', '.join(themes)}")
        selected = [themes[args.theme]]
    elif args.category:
        selected = [t for t in themes.values() if t["category"] == args.category]
    else:
        selected = list(themes.values())
    if not selected:
        raise SystemExit("No themes matched")

    count = 1 if args.test_one else args.count
    width = args.width or PRINT_MODES[args.print_mode][0]
    height = args.height or PRINT_MODES[args.print_mode][1]
    if args.width and args.height:
        pass

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = root / output_root

    jobs = []
    if args.queue:
        queue_path = Path(args.queue)
        if not queue_path.is_absolute():
            queue_path = root / queue_path
        if not queue_path.exists():
            raise SystemExit(f"Queue file not found: {queue_path}")
        queue_data = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}
        queue_items = queue_data.get("jobs", queue_data if isinstance(queue_data, list) else [])
        for item in queue_items:
            if isinstance(item, str):
                if ":" not in item:
                    raise SystemExit(f"Invalid queue item: {item}")
                theme_id, count_text = item.split(":", 1)
                theme_count = int(count_text)
            else:
                theme_id = item["theme"]
                theme_count = int(item.get("count", 1))
            if theme_id not in themes:
                raise SystemExit(f"Unknown theme '{theme_id}'. Available: {', '.join(themes)}")
            if theme_count < 0:
                raise SystemExit(f"Count for '{theme_id}' must be >= 0")
            for variant in range(theme_count):
                seed = (args.seed + variant) if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
                jobs.append((themes[theme_id], seed))
    elif args.interactive:
        print("Available themes:")
        for index, (theme_id, theme) in enumerate(themes.items(), 1):
            print(f"{index}. {theme_id} - {theme['name']}")
        print("Enter theme numbers separated by commas, or enter theme IDs separated by commas.")
        choices = input("Themes: ").strip()
        theme_ids = []
        for choice in choices.split(","):
            choice = choice.strip()
            if choice.isdigit():
                list_ids = list(themes)
                number = int(choice)
                if number < 1 or number > len(list_ids):
                    raise SystemExit(f"Theme number out of range: {number}")
                theme_ids.append(list_ids[number - 1])
            elif choice:
                if choice not in themes:
                    raise SystemExit(f"Unknown theme '{choice}'. Use --list-themes to view IDs.")
                theme_ids.append(choice)
        if not theme_ids:
            raise SystemExit("No themes selected")
        for theme_id in theme_ids:
            theme_count_text = input(f"Unique image count for {theme_id}: ").strip()
            try:
                theme_count = int(theme_count_text)
            except ValueError:
                raise SystemExit(f"Invalid count: {theme_count_text}")
            if theme_count < 0:
                raise SystemExit("Count must be >= 0")
            for variant in range(theme_count):
                seed = (args.seed + variant) if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
                jobs.append((themes[theme_id], seed))
    elif args.test_one:
        theme = selected[0]
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
        jobs.append((theme, seed))
    elif args.theme_counts:
        requested = {}
        for item in args.theme_counts.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise SystemExit("--theme-counts format: theme_id:count,theme_id:count")
            theme_id, count_text = item.split(":", 1)
            theme_id = theme_id.strip()
            try:
                theme_count = int(count_text.strip())
            except ValueError:
                raise SystemExit(f"Invalid count for theme '{theme_id}': {count_text}")
            if theme_id not in themes:
                raise SystemExit(f"Unknown theme '{theme_id}'. Available: {', '.join(themes)}")
            if theme_count < 0:
                raise SystemExit(f"Count for '{theme_id}' must be >= 0")
            requested[theme_id] = theme_count
        for theme_id, theme_count in requested.items():
            for variant in range(theme_count):
                seed = (args.seed + variant) if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
                jobs.append((themes[theme_id], seed))
    elif args.count_per_theme is not None:
        if args.count_per_theme < 0:
            raise SystemExit("--count-per-theme must be >= 0")
        for theme_index, theme in enumerate(selected):
            for variant in range(args.count_per_theme):
                seed = (args.seed + theme_index * args.count_per_theme + variant) if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
                jobs.append((theme, seed))
    else:
        for i in range(count):
            theme = selected[i % len(selected)]
            seed = (args.seed + i) if args.seed is not None else random.SystemRandom().randrange(1, 2**32)
            jobs.append((theme, seed))

    info = detect_hardware()
    print(f"Hardware: RAM={info.ram_gb}GB, system={info.system}, detected_backend={info.backend}, torch={info.torch_version}")
    print(f"Job count: {len(jobs)}, output: {output_root}, resolution: {width}x{height}")
    model_key = args.model or ("sdxl" if platform.system() == "Darwin" else "sd15")
    model_option = get_model_option(model_key)
    steps = args.steps if args.steps is not None else model_option.recommended_steps
    if steps < 1:
        raise SystemExit("--steps must be at least 1")
    if model_key == "flux-schnell" and steps > 4:
        raise SystemExit("FLUX.1-schnell supports 1-4 inference steps; set --steps to a value in that range.")
    selected_model_id = args.model_id or model_option.model_id
    print(f"Model: {selected_model_id} ({model_option.license_name})")
    print(f"Inference steps: {steps}")
    pixel_themes = [theme for theme, _seed in jobs if theme.get("style_family") == "pixel_art"]
    recommended_pixel_models = {
        theme.get("recommended_model", "sd15") for theme in pixel_themes
    }
    if pixel_themes and recommended_pixel_models != {model_key}:
        recommended_labels = ", ".join(
            get_model_option(key).label for key in sorted(recommended_pixel_models)
        )
        print(
            f"PIXEL-ART MODEL NOTE: selected pixel-art themes recommend {recommended_labels}. "
            "SD 1.5 is the practical choice for all three tiers on the tested Arc A530M; "
            "the selected model remains in use. "
            "The tier is styled by prompts and print processing, not separate model weights."
        )
    if model_option.minimum_ram_gb > info.ram_gb:
        print(
            f"WARNING: this app recommends at least {model_option.minimum_ram_gb} GB system RAM "
            f"for {model_key} offloading; "
            f"detected {info.ram_gb:.1f} GB. It may be very slow, page heavily, or fail to load."
        )
    if model_key == "flux-dev":
        print(f"LICENSE WARNING: {model_option.commercial_note} Details: {model_option.license_url}")
    elif model_key == "sd35-medium":
        print(f"LICENSE NOTICE: {model_option.commercial_note} Details: {model_option.license_url}")
    elif model_key in {"flux-schnell", "z-image-turbo"}:
        print(f"MODEL NOTICE: {model_option.commercial_note} Details: {model_option.license_url}")
    elif model_key == "flux-3-klein":
        print(
            f"LICENSE AND ACCESS NOTICE: {model_option.commercial_note} "
            f"Accept the gated terms and authenticate with Hugging Face. Details: {model_option.license_url}"
        )
    elif model_key == "flux2-klein-9b":
        print(
            f"LICENSE AND ACCESS WARNING: {model_option.commercial_note} "
            f"Accept the gated terms and authenticate with Hugging Face. Details: {model_option.license_url}"
        )
    elif model_key in {"flux2-klein-4b", "flux2-klein-base-4b"}:
        print(f"MODEL NOTICE: {model_option.commercial_note} Details: {model_option.license_url}")
    elif model_key == "ideogram-4-fp8":
        print(
            f"LICENSE AND RUNTIME WARNING: {model_option.commercial_note} "
            "The official package must be installed in a separate Python environment. "
            "Its FP8 runner supports CPU, CUDA, or MPS, not Intel XPU/OpenVINO; no hosted "
            "prompt-expansion or moderation services are used. Manually review every output. "
            f"Details: {model_option.license_url}"
        )
    elif model_key == "qwen-image-2.1":
        print(
            f"LICENSE AND HARDWARE NOTICE: {model_option.commercial_note} "
            "The pipeline requires Transformers 5.17 or newer. The model contains a 7B "
            "visual-generation component; CPU/XPU offload may be slow and needs substantial RAM. "
            f"Details: {model_option.license_url}"
        )
    if args.dry_run:
        for theme, seed in jobs:
            print(f"DRY RUN {theme['id']} seed={seed}")
        return

    gen = Generator(backend=args.device, model_id=args.model_id, model_key=model_key,
                    cache_dir=str(root / "models"), openvino_model_dir=args.openvino_model,
                    steps=steps, guidance=args.guidance,
                    scheduler=args.scheduler, style_preset=args.style_preset)
    print(f"Model: {gen.model_id} ({gen.model_option.license_name})  Backend: {gen.backend}")

    success = failed = skipped = 0
    rows = []
    previews = []
    start_total = time.monotonic()
    if jobs and model_key in {"ideogram-4-fp8", "qwen-image-2.1"}:
        try:
            gen.ensure_adapter()
            if gen.adapter is None:
                raise RuntimeError("The selected model adapter was not created.")
            gen.adapter.load()
        except Exception as exc:
            gen.unload()
            model_name = (
                "Ideogram 4"
                if model_key == "ideogram-4-fp8"
                else "Qwen Image 2.1"
            )
            raise SystemExit(
                f"{model_name} setup failed before batch generation; "
                f"no images were generated: {exc}"
            ) from exc

    for i, (theme, seed) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] Generating {theme['name']} seed={seed}")
        try:
            result = gen.generate_one(
                theme=theme, output_dir=output_root / ("etsy" if "etsy" in theme["marketplace"] else "shopee"),
                width=width, height=height, seed=seed,
                extra_prompt=args.extra_prompt, extra_negative=args.extra_negative,
                aquarium_contrast=args.aquarium_contrast,
                variant=i-1, skip_existing=args.skip_existing,
                ai_upscale=args.ai_upscale,
            )
            status = result["status"]
            if status == "skipped": skipped += 1
            else:
                success += 1
                preview = Path(result["Preview path"])
                if preview.exists(): previews.append(preview)
                rows.append(result)
        except Exception as exc:
            failed += 1
            print(f"FAILED {theme['id']}: {exc}", file=sys.stderr)

    gen.unload()
    elapsed = time.monotonic() - start_total
    if previews:
        sheet = root / "outputs" / "contact_sheets" / f"sheet_{int(time.time())}.jpg"
        try:
            contact_sheet(previews, sheet)
            print(f"Contact sheet: {sheet}")
        except Exception:
            pass

    if args.export_csv:
        write_export_csv(root / "outputs" / "catalog.csv", rows)
    if args.export_excel:
        write_export_excel(root / "outputs" / "catalog.xlsx", rows)
    print(f"Success: {success}  Failed: {failed}  Skipped: {skipped}")
    print(f"Model: {gen.model_id}  Backend: {gen.backend}  Total time: {elapsed:.1f}s")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
