from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_pipeline(model_id: str, device: str):
    try:
        import torch
        from ideogram4 import (
            PRESETS,
            Ideogram4Pipeline,
            Ideogram4PipelineConfig,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "ideogram4":
            raise RuntimeError(
                "The official ideogram4 package is not installed in the selected Python "
                "environment. Install it in a separate environment and set IDEOGRAM4_PYTHON."
            ) from exc
        raise

    with contextlib.redirect_stdout(sys.stderr):
        pipeline = Ideogram4Pipeline.from_pretrained(
            config=Ideogram4PipelineConfig(weights_repo=model_id),
            device=device,
            dtype=torch.bfloat16,
        )
    return PRESETS, pipeline


def _generate(presets, pipeline, request: dict) -> None:
    output_path = Path(request["output_path"])
    prompt = request["prompt"]
    steps = int(request["steps"])
    guidance = request.get("guidance")

    kwargs = {
        "height": int(request["height"]),
        "width": int(request["width"]),
        "num_steps": steps,
        "seed": int(request["seed"]),
        "raise_on_caption_issues": True,
    }
    if steps == presets["V4_QUALITY_48"].num_steps and guidance is None:
        preset = presets["V4_QUALITY_48"]
        kwargs.update(
            guidance_schedule=preset.guidance_schedule,
            mu=preset.mu,
            std=preset.std,
        )
    else:
        kwargs["guidance_scale"] = 7.0 if guidance is None else float(guidance)

    with contextlib.redirect_stdout(sys.stderr):
        images = pipeline(prompt, **kwargs)
    if not images:
        raise RuntimeError("The Ideogram 4 pipeline returned no images.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    try:
        presets, pipeline = _load_pipeline(args.model_id, args.device)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _send({"error": f"{type(exc).__name__}: {exc}"})
        return 1

    _send({"ready": True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Worker requests must be JSON objects.")
            _generate(presets, pipeline, request)
            _send({"ok": True})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send({"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
