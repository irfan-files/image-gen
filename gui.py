from __future__ import annotations

import json
import platform
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from model_catalog import MODEL_OPTIONS, MODEL_LABEL_TO_KEY

try:
    import yaml
except ModuleNotFoundError:
    _yaml_error = True
else:
    _yaml_error = False


ROOT = Path(__file__).resolve().parent
THEMES_FILE = ROOT / "themes.yaml"
SETTINGS_FILE = ROOT / "gui_settings.json"
CATEGORIES = {
    "All themes": None,
    "Christmas wall art": "christmas_wall_art",
    "New Year wall art": "new_year_wall_art",
    "Winter landscape": "winter_landscape",
    "Aquarium": "aquarium",
}


def load_themes() -> dict[str, dict]:
    if _yaml_error:
        raise RuntimeError(
            "PyYAML belum terpasang di Python environment yang menjalankan GUI.\n\n"
            "Jalankan start_gui.bat; launcher akan memasang dependency GUI yang hilang."
        )
    data = yaml.safe_load(THEMES_FILE.read_text(encoding="utf-8"))
    themes = data.get("themes", [])
    return {theme["id"]: theme for theme in themes}


class SeasonalArtGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Seasonal Art Factory")
        self.root.geometry("1120x820")
        self.root.minsize(900, 600)
        self.themes = load_themes()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.theme_vars: dict[str, tk.BooleanVar] = {}
        self.count_vars: dict[str, tk.StringVar] = {}
        self._load_settings()
        self._build_ui()
        self._render_themes()
        self.root.after(100, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self.saved_counts = data.get("counts", {})
        self.saved_selected = set(data.get("selected", []))
        self.saved_category = data.get("category", "All themes")
        self.saved_backend = data.get("backend", "xpu")
        self.saved_model = data.get("model", "sdxl" if platform.system() == "Darwin" else "sd15")
        default_steps = MODEL_OPTIONS.get(self.saved_model, MODEL_OPTIONS["sd15"]).recommended_steps
        default_steps = default_steps if self.saved_model in {"flux-schnell", "z-image-turbo"} else 20
        try:
            self.saved_steps = int(data.get("steps", default_steps))
        except (TypeError, ValueError):
            self.saved_steps = default_steps
        if self.saved_model == "flux-schnell" and not 1 <= self.saved_steps <= 4:
            self.saved_steps = MODEL_OPTIONS["flux-schnell"].recommended_steps
        self.saved_width = data.get("width", 6000)
        self.saved_height = data.get("height", 3000)
        self.saved_upscale = data.get("ai_upscale", True)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(4, weight=1)

        title = ttk.Label(
            self.root, text="Seasonal Art Factory",
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        settings = ttk.LabelFrame(self.root, text="Batch settings")
        settings.grid(row=1, column=0, sticky="ew", padx=14, pady=8)
        for col in range(12):
            settings.columnconfigure(col, weight=0)

        ttk.Label(settings, text="Model").grid(row=0, column=0, padx=(8, 4), pady=8)
        model_labels = [option.label for option in MODEL_OPTIONS.values()]
        saved_model_label = MODEL_OPTIONS.get(self.saved_model, MODEL_OPTIONS["sd15"]).label
        self.model_var = tk.StringVar(value=saved_model_label)
        self.model_box = ttk.Combobox(
            settings, textvariable=self.model_var, state="readonly",
            values=model_labels, width=48,
        )
        self.model_box.grid(row=0, column=1, columnspan=3, sticky="w", padx=4, pady=8)
        self.model_box.bind("<<ComboboxSelected>>", self._model_changed)
        self._current_model_key = self.saved_model

        ttk.Label(settings, text="Category").grid(row=0, column=4, padx=(12, 4), pady=8)
        self.category_var = tk.StringVar(value=self.saved_category)
        category = ttk.Combobox(
            settings, textvariable=self.category_var, state="readonly",
            values=list(CATEGORIES), width=20,
        )
        category.grid(row=0, column=5, padx=4, pady=8)
        category.bind("<<ComboboxSelected>>", lambda _event: self._render_themes())

        ttk.Label(settings, text="Backend").grid(row=1, column=0, padx=(8, 4), pady=8)
        self.backend_var = tk.StringVar(value=self.saved_backend)
        ttk.Combobox(
            settings, textvariable=self.backend_var, state="readonly",
            values=["xpu", "auto", "cpu"], width=9,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=8)

        ttk.Label(settings, text="Inference steps").grid(row=1, column=2, padx=(12, 4), pady=8)
        self.steps_var = tk.StringVar(value=str(self.saved_steps))
        ttk.Spinbox(settings, from_=1, to=100, textvariable=self.steps_var, width=6).grid(
            row=1, column=3, padx=4, pady=8
        )

        ttk.Label(settings, text="Width").grid(row=2, column=0, padx=(8, 4), pady=8)
        self.width_var = tk.StringVar(value=str(self.saved_width))
        ttk.Entry(settings, textvariable=self.width_var, width=9).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(settings, text="Height").grid(row=2, column=2, padx=(12, 4), pady=8)
        self.height_var = tk.StringVar(value=str(self.saved_height))
        ttk.Entry(settings, textvariable=self.height_var, width=9).grid(row=2, column=3, sticky="w", padx=4)

        self.upscale_var = tk.BooleanVar(value=self.saved_upscale)
        ttk.Checkbutton(
            settings, text="Real-ESRGAN x4 AI upscale (requires model weights)",
            variable=self.upscale_var,
        ).grid(row=2, column=4, columnspan=5, sticky="w", padx=(12, 4), pady=8)

        self.model_note_var = tk.StringVar()
        ttk.Label(
            settings, textvariable=self.model_note_var, wraplength=950,
            foreground="#8a3b12",
        ).grid(row=3, column=0, columnspan=9, sticky="w", padx=8, pady=(0, 6))
        self._model_changed()

        actions = ttk.Frame(settings)
        actions.grid(row=4, column=0, columnspan=9, sticky="ew", padx=8, pady=(2, 8))
        ttk.Label(actions, text="Bulk-select themes, then set their count:").pack(side="left")
        self.bulk_count_var = tk.StringVar(value="1")
        ttk.Spinbox(actions, from_=0, to=10000, textvariable=self.bulk_count_var, width=7).pack(
            side="left", padx=6
        )
        ttk.Button(actions, text="Apply count", command=self._apply_bulk_count).pack(side="left")
        ttk.Button(actions, text="Select visible", command=self._select_visible).pack(side="left", padx=(12, 4))
        ttk.Button(actions, text="Clear selection", command=self._clear_selection).pack(side="left", padx=4)
        ttk.Label(actions, text="Any positive count is queued.").pack(side="left", padx=10)

        theme_box = ttk.LabelFrame(self.root, text="Select themes and set images per theme")
        theme_box.grid(row=2, column=0, sticky="nsew", padx=14, pady=8)
        theme_box.columnconfigure(0, weight=1)
        theme_box.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(theme_box, highlightthickness=0)
        scrollbar = ttk.Scrollbar(theme_box, orient="vertical", command=self.canvas.yview)
        self.rows_frame = ttk.Frame(self.canvas)
        self.rows_window = self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.rows_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.rows_window, width=event.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)

        footer = ttk.Frame(self.root)
        footer.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 8))
        footer.columnconfigure(0, weight=1)
        self.summary_var = tk.StringVar(value="Choose themes and set a count for each.")
        ttk.Label(footer, textvariable=self.summary_var).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=6)
        self.start_button = ttk.Button(footer, text="Start generation", command=self._start)
        self.start_button.grid(row=0, column=1, padx=4)
        self.cancel_button = ttk.Button(footer, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=0, column=2, padx=4)

        log_frame = ttk.LabelFrame(self.root, text="Progress and messages")
        log_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 12))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def _render_themes(self) -> None:
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self.theme_vars.clear()
        self.count_vars.clear()
        ttk.Label(self.rows_frame, text="Bulk", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        ttk.Label(self.rows_frame, text="Theme", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=8, pady=4
        )
        ttk.Label(self.rows_frame, text="Category", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=2, sticky="w", padx=8, pady=4
        )
        ttk.Label(self.rows_frame, text="Images (0 = skip)", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=3, sticky="w", padx=8, pady=4
        )

        category_filter = CATEGORIES.get(self.category_var.get())
        visible = [
            (theme_id, theme) for theme_id, theme in self.themes.items()
            if category_filter is None or theme.get("category") == category_filter
        ]
        for row, (theme_id, theme) in enumerate(visible, start=1):
            selected = theme_id in self.saved_selected
            count = self.saved_counts.get(theme_id, 0)
            checked = tk.BooleanVar(value=selected or int(count or 0) > 0)
            count_var = tk.StringVar(value=str(count))
            self.theme_vars[theme_id] = checked
            self.count_vars[theme_id] = count_var
            count_var.trace_add("write", lambda *_args: self._update_summary())
            theme_label = f"{theme['name']}  ({theme_id})"
            if theme.get("style_family") == "pixel_art":
                tier = theme.get("pixel_art_era", "pixel art")
                model = theme.get("recommended_model", "sd15").upper()
                theme_label += f"  — {tier}, {model} recommended"
            ttk.Checkbutton(self.rows_frame, variable=checked).grid(row=row, column=0, padx=8, pady=2)
            ttk.Label(
                self.rows_frame, text=theme_label,
            ).grid(row=row, column=1, sticky="w", padx=8, pady=2)
            ttk.Label(self.rows_frame, text=theme.get("category", "")).grid(
                row=row, column=2, sticky="w", padx=8, pady=2
            )
            ttk.Spinbox(
                self.rows_frame, from_=0, to=10000, textvariable=count_var, width=9,
            ).grid(row=row, column=3, sticky="w", padx=8, pady=2)
        self._update_summary()

    def _mousewheel(self, event: tk.Event) -> None:
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _update_summary(self) -> None:
        total = 0
        selected = 0
        for theme_id, variable in self.theme_vars.items():
            try:
                count = int(self.count_vars[theme_id].get())
            except ValueError:
                count = 0
            if count > 0:
                selected += 1
                total += count
        self.summary_var.set(f"{selected} selected themes  |  {total} images queued")

    def _apply_bulk_count(self) -> None:
        try:
            count = int(self.bulk_count_var.get())
            if count < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid count", "Count must be a whole number greater than or equal to zero.")
            return
        for theme_id, variable in self.theme_vars.items():
            if variable.get():
                self.count_vars[theme_id].set(str(count))
        self._update_summary()

    def _select_visible(self) -> None:
        for variable in self.theme_vars.values():
            variable.set(True)
        self._update_summary()

    def _clear_selection(self) -> None:
        for variable in self.theme_vars.values():
            variable.set(False)
        self._update_summary()

    def _collect_counts(self) -> dict[str, int]:
        counts = {}
        for theme_id in self.count_vars:
            try:
                count = int(self.count_vars[theme_id].get())
            except ValueError:
                raise ValueError(f"Image count for {theme_id} must be a whole number.")
            if count < 0:
                raise ValueError(f"Image count for {theme_id} cannot be negative.")
            if count > 0:
                counts[theme_id] = count
        return counts

    def _save_settings(self, counts: dict[str, int]) -> None:
        data = {
            "counts": {**self.saved_counts, **counts},
            "selected": [theme_id for theme_id, var in self.theme_vars.items() if var.get()],
            "category": self.category_var.get(),
            "backend": self.backend_var.get(),
            "model": MODEL_LABEL_TO_KEY[self.model_var.get()],
            "steps": int(self.steps_var.get()),
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "ai_upscale": self.upscale_var.get(),
        }
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.saved_counts = data["counts"]
        self.saved_selected = set(data["selected"])

    def _start(self) -> None:
        if self.process is not None:
            return
        try:
            counts = self._collect_counts()
            steps = int(self.steps_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if not counts:
                raise ValueError("Select at least one theme and enter a count above zero.")
            if steps < 1 or width < 64 or height < 64:
                raise ValueError("Steps must be positive and image dimensions must be at least 64.")
        except ValueError as exc:
            messagebox.showerror("Check batch settings", str(exc))
            return

        try:
            self._save_settings(counts)
        except OSError as exc:
            messagebox.showerror("Cannot save settings", str(exc))
            return

        model_key = MODEL_LABEL_TO_KEY[self.model_var.get()]
        model_option = MODEL_OPTIONS[model_key]
        pixel_themes = [
            self.themes[theme_id] for theme_id in counts
            if self.themes[theme_id].get("style_family") == "pixel_art"
        ]
        recommended_pixel_models = {
            theme.get("recommended_model", "sd15") for theme in pixel_themes
        }
        if pixel_themes and recommended_pixel_models != {model_key}:
            recommended_labels = ", ".join(
                MODEL_OPTIONS[key].label for key in sorted(recommended_pixel_models)
            )
            if not messagebox.askyesno(
                "Pixel-art model recommendation",
                f"The selected pixel-art themes recommend: {recommended_labels}. "
                "For the tested Intel Arc A530M, SD 1.5 is the practical choice across "
                "8-bit, 16-bit, and 32-bit styles; each tier is controlled by its prompt "
                "and crisp print processing. "
                "The selected model will still be used. Continue?",
            ):
                return
        if model_key == "flux-dev":
            if not messagebox.askyesno(
                "FLUX.1-dev license warning",
                f"{model_option.commercial_note}\n\n"
                "This model requires access approval on Hugging Face and a local license review. "
                "Continue only if your intended use is permitted.",
            ):
                return
        elif model_key == "sd35-medium":
            if not messagebox.askyesno(
                "Stable Diffusion 3.5 Medium license",
                f"{model_option.commercial_note}\n\nContinue only if you comply with the Stability AI Community License.",
            ):
                return
        elif model_key == "z-image-turbo":
            if not messagebox.askyesno(
                "Z-Image-Turbo hardware notice",
                f"{model_option.commercial_note}\n\n"
                "On an Intel Arc A530M with 4 GB VRAM, generation may be extremely slow. Continue?",
            ):
                return

        theme_counts = ",".join(f"{theme_id}:{count}" for theme_id, count in counts.items())
        command = [
            sys.executable, "-u", str(ROOT / "generate.py"),
            "--model", model_key,
            "--theme-counts", theme_counts,
            "--device", self.backend_var.get(),
            "--width", str(width), "--height", str(height),
            "--steps", str(steps),
            "--export-csv",
        ]
        if self.upscale_var.get():
            command.append("--ai-upscale")
        else:
            command.append("--no-ai-upscale")

        self._append_log(f"Starting batch: {sum(counts.values())} images across {len(counts)} themes")
        self._append_log(f"Interpreter: {sys.executable}")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(value=0, maximum=sum(counts.values()))
        threading.Thread(target=self._run_process, args=(command,), daemon=True).start()

    def _model_changed(self, _event: tk.Event | None = None) -> None:
        model_key = MODEL_LABEL_TO_KEY.get(self.model_var.get(), "sd15")
        option = MODEL_OPTIONS[model_key]
        if model_key != self._current_model_key:
            self.steps_var.set(str(option.recommended_steps))
            self._current_model_key = model_key
        ram = f"App-recommended system RAM for offloading: {option.minimum_ram_gb} GB."
        self.model_note_var.set(
            f"{option.license_name}. {option.commercial_note} {ram} "
            f"Model card: {option.license_url}"
        )

    def _run_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("line", line.rstrip()))
            code = self.process.wait()
            self.events.put(("finished", code))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    line = str(value)
                    self._append_log(line)
                    match = re.search(r"\[(\d+)/(\d+)\]", line)
                    if match:
                        self.progress.configure(value=int(match.group(1)), maximum=int(match.group(2)))
                elif kind == "finished":
                    code = int(value)
                    self.process = None
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if code == 0:
                        self._append_log("Batch completed.")
                    else:
                        self._append_log(f"Batch ended with exit code {code}. See messages above.")
                elif kind == "error":
                    self.process = None
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self._append_log(f"ERROR: {value}")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _cancel(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._append_log("Cancel requested. Current image may not be saved; completed images remain on disk.")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Generation running", "Cancel generation and close the app?"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        SeasonalArtGUI(root)
    except Exception as exc:
        messagebox.showerror("Seasonal Art Factory", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
