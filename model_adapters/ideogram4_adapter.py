from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from .base import BaseGenerator


SUPPORTED_BACKENDS = {"cpu", "cuda", "mps"}


def build_ideogram_caption(prompt: str, negative_prompt: str = "") -> str:
    description = prompt.strip()
    if negative_prompt.strip():
        description += f"\nAvoid: {negative_prompt.strip()}"
    caption = {
        "high_level_description": description,
        "compositional_deconstruction": {
            "background": "A full-bleed background supporting the described scene.",
            "elements": [{"type": "obj", "desc": prompt.strip()}],
        },
    }
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


class Ideogram4Adapter(BaseGenerator):
    """Runs the official Ideogram package in its own Python environment."""

    def __init__(self, model_id: str, backend: str, cache_dir: str):
        super().__init__(model_id, backend, cache_dir)
        self.process: subprocess.Popen | None = None

    def _python_executable(self) -> str:
        executable = os.environ.get("IDEOGRAM4_PYTHON", "").strip()
        if not executable:
            executable_path = (
                Path(__file__).resolve().parents[1]
                / ".ideogram4"
                / "Scripts"
                / "python.exe"
            )
            if not executable_path.is_file():
                raise RuntimeError(
                    "The separate Ideogram 4 environment was not found. Install it at "
                    f"{executable_path.parent.parent} or set IDEOGRAM4_PYTHON "
                    "to its Python executable."
                )
        else:
            executable_path = Path(executable)
        if not executable_path.is_file():
            raise RuntimeError(
                f"Ideogram 4 Python interpreter does not exist: {executable_path}. "
                "Set IDEOGRAM4_PYTHON to the Python executable in the separate environment."
            )
        if executable_path.resolve() == Path(sys.executable).resolve():
            raise RuntimeError(
                "IDEOGRAM4_PYTHON must point to a separate environment, not the app's "
                "Python interpreter."
            )
        return str(executable_path.resolve())

    def _require_huggingface_auth(self, executable: str) -> None:
        try:
            result = subprocess.run(
                [
                    executable,
                    "-c",
                    "from huggingface_hub import get_token; "
                    "print(bool(get_token()))",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Timed out checking Hugging Face authentication in the Ideogram environment."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "Could not check Hugging Face authentication in the Ideogram environment: "
                f"{result.stderr.strip() or 'the authentication check failed'}"
            )
        if result.stdout.strip().lower() != "true":
            raise RuntimeError(
                "No Hugging Face token is configured. Accept the model terms, then run "
                "'.\\.ideogram4\\Scripts\\hf.exe auth login' from the project folder."
            )

    def _start_worker(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"The official Ideogram 4 FP8 runner does not support backend "
                f"'{self.backend}'. Choose CPU, CUDA, or MPS; Intel XPU/OpenVINO are unsupported."
            )

        executable = self._python_executable()
        self._require_huggingface_auth(executable)
        worker_path = Path(__file__).with_name("ideogram4_worker.py")
        environment = os.environ.copy()
        environment["HF_HUB_CACHE"] = str(Path(self.cache_dir).resolve())
        self.process = subprocess.Popen(
            [
                executable,
                "-u",
                str(worker_path),
                "--model-id",
                self.model_id,
                "--device",
                self.backend,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        try:
            response = self._read_response()
        except Exception:
            self._stop_worker()
            raise
        if response.get("ready") is not True:
            error = response.get("error", "Worker did not report readiness.")
            self._stop_worker()
            raise RuntimeError(f"Ideogram 4 worker failed to load: {error}")
        self.loaded = True

    def _read_response(self) -> dict:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("Ideogram 4 worker is not running.")
        line = process.stdout.readline()
        if not line:
            return_code = process.poll()
            raise RuntimeError(
                "Ideogram 4 worker exited without a response"
                + (f" (exit code {return_code})." if return_code is not None else ".")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ideogram 4 worker returned invalid protocol data: {line.rstrip()!r}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("Ideogram 4 worker returned a non-object response.")
        if response.get("error"):
            raise RuntimeError(f"Ideogram 4 worker error: {response['error']}")
        return response

    def _stop_worker(self) -> None:
        process = self.process
        self.process = None
        self.loaded = False
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()

    def load(self) -> None:
        if self.loaded and self.process is not None and self.process.poll() is None:
            return
        self._stop_worker()
        self._start_worker()

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
        if scheduler.lower() not in {"dpmsolver", "default"}:
            raise ValueError(
                "Ideogram 4 uses its built-in sampler presets; "
                f"the scheduler '{scheduler}' is not supported."
            )
        if not self.loaded or self.process is None or self.process.poll() is not None:
            self.load()
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("Ideogram 4 worker is not available.")

        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
                output_path = Path(output_file.name)
            request = {
                "prompt": build_ideogram_caption(prompt, negative_prompt),
                "output_path": str(output_path),
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
            }
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = self._read_response()
            if response.get("ok") is not True:
                raise RuntimeError("Ideogram 4 worker did not confirm image generation.")
            if not output_path.is_file():
                raise RuntimeError("Ideogram 4 worker reported success but saved no image.")
            with Image.open(output_path) as image:
                return image.convert("RGB")
        finally:
            if output_path is not None:
                try:
                    output_path.unlink()
                except FileNotFoundError:
                    pass

    def unload(self) -> None:
        self._stop_worker()
