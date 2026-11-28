# Seasonal Art Factory

Production-oriented local AI artwork batch generator for Etsy, Shopee, wall art, and aquarium backgrounds.

## Model selection

| Platform | Default | Native resolution | Acceleration | Notes |
|---|---|---:|---|---|
| macOS Apple Silicon | `stabilityai/stable-diffusion-xl-base-1.0` | 1536×768 | PyTorch MPS | 32 GB unified memory; try 1536, 1344, 1216, 1024; SDXL license/Rail terms must be reviewed |
| Windows Intel Arc A530M | `runwayml/stable-diffusion-v1-5` | 512×256 to 768×384 | OpenVINO, verified XPU, or CPU | Recommended for 4 GB VRAM and all pixel-art tiers (8/16/32-bit style) |
| Optional | `stabilityai/stable-diffusion-xl-base-1.0` | 640×320 on Arc; larger on suitable hardware | Diffusers | FP32 VAE with layer-by-layer RAM offload; slower on 4 GB VRAM |
| Optional | `stabilityai/stable-diffusion-3.5-medium` | 512×256 on Arc | Diffusers `StableDiffusion3Pipeline` | Stability AI Community License; below USD 1M annual revenue commercial use is permitted subject to the license. Review current terms |
| Optional | `black-forest-labs/FLUX.1-dev` | 512×256 on Arc | Diffusers `FluxPipeline` | Non-commercial license. Do not use for Etsy/Shopee sales without a separate commercial license |
| Optional | `black-forest-labs/FLUX.1-schnell` | 512×256 on Arc; 4 steps by default | Diffusers `FluxPipeline` | Apache 2.0. Still 12B parameters; RAM offload on a 4 GB GPU is expected to be slow |
| Optional | `Tongyi-MAI/Z-Image-Turbo` | 512×256 on Arc; 9 steps by default | Diffusers `ZImagePipeline` | Apache 2.0. 6B parameters; model card targets GPUs with about 16 GB VRAM, so 4 GB Arc requires slow RAM offload |
| Optional | `black-forest-labs/flux-3-image-klein` | 512×256 on Arc; 16 steps by default | Diffusers `ModularPipeline` | Gated FLUX Kommunity License. Model card reports 31 GB peak GPU memory without prompt upsampling; its upsampler adds about 28 GB on an 80 GB GPU |
| Optional | `black-forest-labs/FLUX.2-klein-9B` | 512×256 on Arc; 4 steps by default | Diffusers `Flux2KleinPipeline` | Gated non-commercial license; do not use for Etsy/Shopee sales. Model card reports about 29 GB VRAM and requires safety filters or manual review |
| Optional | `black-forest-labs/FLUX.2-klein-4B` | 512×256 on Arc; 4 steps by default | Diffusers `Flux2KleinPipeline` | Apache 2.0, including commercial use. Model card reports about 13 GB VRAM; safety filters and manual review are encouraged |
| Optional | `black-forest-labs/FLUX.2-klein-base-4B` | 512×256 on Arc; 50 steps by default | Diffusers `Flux2KleinPipeline` | Apache 2.0, including commercial use. Model card reports about 13 GB VRAM; safety filters and manual review are encouraged |
| Optional | `ideogram-ai/ideogram-4-fp8` | 1024×1024 native; 48 steps by default | Official `ideogram4` package in an isolated Python worker | Gated non-commercial agreement; no Etsy/Shopee sales. FP8 supports CPU/CUDA/MPS, not Intel XPU/OpenVINO |
| Optional | `Qwen/Qwen-Image-2.1` | 40 steps by default; 7B visual-generation component | Diffusers `QwenImage21Pipeline` | Qwen Research License is non-commercial; commercial output requires a separate Qwen license. App recommends 64 GB RAM for offloading |

The GUI's model selector starts with SD 1.5. FLUX.1-dev, SD 3.5 Medium, FLUX 3 Image Klein, and FLUX.2 Klein 9B display license confirmations before generation. Z-Image-Turbo, FLUX.2 Klein 4B, FLUX.2 Klein Base 4B, and the Klein 9B display hardware warnings before generation. FLUX.1-dev requires Hugging Face model access approval and a separate commercial license for commercial product sales. FLUX 3 Image Klein is gated and requires accepting the model terms and authenticating with Hugging Face; commercial output generation is limited to qualifying users below US$5M gross annualized revenue, subject to the full license and acceptable-use terms. FLUX.2 Klein 9B is gated, requires Hugging Face approval, is non-commercial only, and requires safety filters or manual review. Both FLUX.2 Klein 4B presets are Apache 2.0. The GUI cannot grant model access or licenses.

On the current Arc A530M setup, SD 1.5 XPU generation was tested. A one-step FLUX.1-dev XPU smoke test also completed at 512×256 after applying a float32 fallback for FLUX positional embeddings, which otherwise request unsupported float64 on this PyTorch XPU device. That single denoising step took about four minutes with sequential CPU/RAM offload and classifier-free guidance, so FLUX is not practical for large batches on this hardware. FLUX.1-dev is a 12B-parameter model and recommends substantially more memory; SD 3.5 Medium also has a significantly larger memory footprint than SD 1.5. The app warns when detected system RAM is below the model's configured recommendation, but offloading cannot guarantee acceptable speed or prevent all out-of-memory errors.

For FLUX.1-dev, a non-empty negative prompt enables true classifier-free guidance with `true_cfg_scale=2.0`; without a negative prompt, that extra guidance is omitted.

FLUX.1-schnell uses guidance scale 0, does not use negative prompts, and limits inference to 1-4 steps (4 by default). Z-Image-Turbo uses guidance scale 0 and 9 steps by default. FLUX 3 Image Klein uses its distilled no-guidance workflow and runs the model's prompt upsampler automatically. FLUX.2 Klein 9B and 4B default to 4 steps and guidance 1.0; Klein Base 4B defaults to 50 steps and guidance 4.0. FLUX 3 and FLUX.2 Klein models do not use the app's negative prompt. These settings follow the respective model cards; you can change steps in the GUI or with `--steps`.

Models are loaded one at a time. Diffusers, Transformers, and PyTorch XPU versions must support the selected pipeline. The install requirements need Diffusers 0.43 or newer for Z-Image-Turbo and FLUX.2 Klein, and Transformers 4.57 or newer for FLUX 3 Image Klein.

Ideogram 4 FP8 is intentionally isolated from the app's PyTorch environment because the official `ideogram4` package requires PyTorch 2.11 or newer and does not support Intel XPU. It uses the official 48-step quality preset by default; the generic `--scheduler` setting does not apply. The app wraps its existing positive and negative prompts in Ideogram's structured JSON caption locally; it does not call hosted magic-prompt or moderation services. This means no prompts are sent to those services, but no remote safety screening is performed either. Review every generated image manually.

Qwen Image 2.1 uses `QwenImage21Pipeline`, 40 steps by default, BF16 weights where available, and sequential/model CPU offload on constrained accelerators. It requires Transformers 5.17 or newer (the Windows installer pins a compatible PyTorch XPU build separately). The model is under the Qwen Research License Agreement: commercial use requires a separate license from Qwen; the GUI asks for confirmation before commercial workflows. The model card lists a 7B visual-generation component, so the app recommends 64 GB system RAM for offloading; this is a project recommendation, not a guarantee of inference success.

### Long prompts and wallpaper-only output

Generated prompts explicitly request a flat, full-bleed standalone artwork. Aquarium themes request the underwater scene itself, without a display frame or equipment, and their negative prompts exclude physical tanks and aquarium hardware. All themes also exclude framed-print/product mockups, staged rooms, people, figurines, and statues. These prompt constraints reduce presentation mistakes but cannot guarantee every model result; review previews before printing or listing.

The app no longer relies on CLIP silently truncating long prompts at 77 tokens. SD 1.5 encodes longer text in 75-token CLIP chunks; SDXL chunks for both CLIP tokenizers. FLUX and SD 3.5 receive a short, CLIP-safe subject/style anchor and the complete positive and negative prompt through their T5 encoder. Z-Image receives the complete prompt through its own sequence encoder. The app expands sequence limits when needed, up to 512 tokens, and reports an error instead of silently cutting text beyond the model limit. FLUX.1-schnell does not use negative prompts by design. Existing generated files are unchanged; these corrections apply to new generations.

## Architecture

- `hardware.py`: CPU/RAM/GPU/backend detection.
- `model_adapters/`: modular Diffusers, OpenVINO, and future adapters.
- `prompt_builder.py`: controlled themed prompt and negative-prompt composition.
- `generator.py`: queue, reproducible seeds, OOM fallback, atomic PNG writes, metadata and resume behavior.
- `quality_control.py`: dimensions, mode, integrity, blur, clipping, pHash duplicate detection, contact sheets.
- `upscale.py`: tiled PyTorch Real-ESRGAN, Lanczos framing, and tier-specific pixel-grid rendering.
- `metadata.py`: atomic JSON, CSV, and Excel exports.
- `themes.yaml`: validated Christmas, New Year, winter, and aquarium themes, including 14 pixel-art designs.

## macOS setup

```bash
cd seasonal-art-factory
bash install_macos.sh
source .venv/bin/activate
huggingface-cli login
python generate.py --test-one --theme vintage_winter_cabin --device mps
```

Official model: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0  
License/status: review the model card and CreativeML Open RAIL++-M terms for the exact use case.

For SDXL on MPS, the app starts native at 1536×768 only when width is 6000 or large-print; it retries lower resolutions on memory failure. MPS support varies by PyTorch and macOS versions.

## Windows setup

Recommended install for the tested Intel XPU stack:

```powershell
cd seasonal-art-factory
.\install_windows.ps1
.\.venv\Scripts\Activate.ps1
.\start_gui.bat
```

For verified XPU, install a PyTorch build reported compatible with your Intel driver and Arc GPU. The app detects `torch.xpu`; it will not assume CUDA. The implementation has been tested with PyTorch `2.8.0+xpu` on an Intel Arc A530M. The Arc GPU holds 4 GB VRAM; large models use sequential CPU/RAM offload.

For gated model repositories, accept the model terms on Hugging Face and authenticate in the same environment with `hf auth login`. The model adapter reports a specific access error if approval or authentication is missing.

#### Optional Ideogram 4 FP8 environment

Ideogram 4 FP8 is governed by a non-commercial agreement; do not use it to create Etsy/Shopee listings or other commercial work. Accept the gated model terms first. Install the official package in its own environment rather than changing the app's XPU environment:

```powershell
py -3.12 -m venv .ideogram4
.\.ideogram4\Scripts\python.exe -m pip install git+https://github.com/ideogram-oss/ideogram4.git
.\.ideogram4\Scripts\hf.exe auth login
.\start_gui.bat
```

The app auto-detects the interpreter at `.ideogram4\Scripts\python.exe`. If you installed the environment elsewhere, set `IDEOGRAM4_PYTHON` to its `python.exe` path before launching the GUI or CLI. The official FP8 runner supports CPU, CUDA, and MPS only; Intel XPU/OpenVINO are unsupported. Choose a supported backend in the app. CPU execution may need substantially more than the app's 32 GB RAM recommendation. The isolated worker loads `ideogram-ai/ideogram-4-fp8` once and reuses it for the batch. Hugging Face model access approval/authentication is still required. The app uses a local JSON caption wrapper and does not enable the official runner's hosted magic-prompt or Hive moderation APIs; human review is required. If the environment is missing or fails to load, the app aborts before starting the batch instead of marking every image failed.

### GUI batch generation

Double-click `start_gui.bat`. It launches the GUI using the project-local `.venv\Scripts\python.exe`, and installs the small GUI YAML dependency into that same environment if it is missing. In the GUI, enter an image count for each theme; rows with a count of zero are skipped. Select a backend, resolution, inference steps, and whether to run Real-ESRGAN, then click **Start generation**.

The GUI saves selected counts and settings in `gui_settings.json`. The generated batch runs sequentially and progress is shown in the GUI. A positive image count is used regardless of the row's checkbox; checkboxes are for bulk-selecting rows before using **Apply count**.

Select one of the model presets at the top of the GUI: SD 1.5 (recommended for the Arc), SDXL, FLUX.1-dev, SD 3.5 Medium, FLUX.1-schnell, Z-Image-Turbo, FLUX 3 Image Klein, FLUX.2 Klein 9B, FLUX.2 Klein 4B, FLUX.2 Klein Base 4B, Ideogram 4 FP8, or Qwen Image 2.1. Each preset selects its model-card step default (Schnell 4, Z-Image 9, FLUX 3 Klein 16, FLUX.2 Klein 9B/4B 4, Base 4B 50, Ideogram 4 FP8 48, Qwen Image 2.1 40); it can be adjusted as appropriate. Klein models are integrated for text-to-image generation; reference-image editing is not exposed in this batch workflow. The CLI equivalent is `--model sd15`, `--model sdxl`, `--model flux-dev`, `--model sd35-medium`, `--model flux-schnell`, `--model z-image-turbo`, `--model flux-3-klein`, `--model flux2-klein-9b`, `--model flux2-klein-4b`, `--model flux2-klein-base-4b`, `--model ideogram-4-fp8`, or `--model qwen-image-2.1`.

### Pixel-art collections and product strategy

The theme list includes 14 pixel-art concepts: four Christmas wall prints, four year-neutral New Year scenes, and six aquarium backgrounds, spanning 8-bit, 16-bit, and 32-bit-inspired styles. The GUI labels each pixel theme with its style tier and recommended model. Filter by **Christmas wall art**, **New Year wall art**, or **Aquarium**, then enter a separate image count for each theme as usual. The same themes can be selected from the CLI with `--category new_year_wall_art` or `--category aquarium`.

**Recommended model for all three tiers: Stable Diffusion 1.5 (`sd15`).** On the tested 4 GB Arc A530M, it is the practical model already validated on XPU; changing from 8-bit to 16-bit or 32-bit art is a style/detail change, not a need for a larger diffusion model. The tier names are artistic presets, not claims of exact console-era emulation or hardware color limits. Prompts, palette size, and pixel-grid density vary by tier: 8-bit-inspired uses 16 colors and a coarse grid, 16-bit-inspired uses 32 colors and a medium grid, and 32-bit-inspired uses 64 colors and a finer grid. For the standard 6000×3000 master, these grids are 96×48, 192×96, and 288×144 cells respectively.

Pixel-art outputs deliberately bypass Real-ESRGAN: the print master is cropped to the requested aspect ratio, reduced to the tier's grid and palette, then enlarged with nearest-neighbor interpolation. This preserves hard pixel edges at sizes such as 6000×3000, but does not create new source detail or make an image suitable for every print size without review. The GUI may still have the AI-upscale option selected; it is intentionally ignored for pixel-art themes. The recommendation is based on the project's tested hardware and current model setup, not a benchmark proving SD 1.5 artistically superior to every model. Larger presets remain selectable, but are less practical on this GPU and are not needed to change the pixel-art tier.

The product concepts are hypotheses, not sales forecasts. Holiday prints are seasonal and should be tested as coordinated bundles and year-neutral designs; aquarium backgrounds are less seasonal and can support evergreen listings, especially when the center remains visually open. Before investing in a large batch, compare marketplace search suggestions and competing listings, test a small set of thumbnails/titles/tags, and track clicks and conversions separately on Etsy and Shopee. No live sales-volume or ranking data was available to validate demand for these specific concepts. Use original generic motifs, check each marketplace's current AI-content and listing rules, and verify any third-party model or LoRA license for commercial use; no pixel-art LoRA is required or bundled by this workflow.

The error `ModuleNotFoundError: No module named 'yaml'` means PyYAML is absent from the specific Python interpreter used to run the application. Installing PyYAML into a global Python does not install it in `.venv`. Use `start_gui.bat`; for a manual repair, run `.venv\Scripts\python.exe -m pip install PyYAML`.

OpenVINO workflow:

1. Install `openvino`, `optimum-intel`, and Diffusers dependencies.
2. Export a verified Stable Diffusion 1.5 pipeline with Optimum Intel to a local directory.
3. Run:

```powershell
python generate.py --test-one --theme frozen_crystal_cave --device openvino --openvino-model C:\models\sd15-openvino
```

Official SD1.5 model: https://huggingface.co/runwayml/stable-diffusion-v1-5  
License/status: review the CreativeML Open RAIL-M model card. Use an export compatible with the installed OpenVINO runtime and GPU drivers.

The OpenVINO adapter validates the exported directory and dependency before loading. It will not claim success if GPU compilation fails.

## Generation examples

```bash
python generate.py --theme vintage_winter_cabin --count 10 --device auto
python generate.py --category aquarium --count 50 --width 6000 --height 3000
python generate.py --theme frozen_crystal_cave --count 5 --print-mode large_print
python generate.py --input themes.yaml --resume
python generate.py --theme snowy_pine_forest --count 100 --dry-run
python generate.py --test-one --theme winter_moonlight_landscape --device mps
```

## Manual dynamic batch counts

List theme IDs:

```powershell
python generate.py --list-themes
```

Choose themes and counts interactively:

```powershell
python generate.py --interactive --device xpu --width 6000 --height 3000
```

Generate a fixed count for every selected theme:

```powershell
python generate.py --count-per-theme 10 --device xpu
python generate.py --category aquarium --count-per-theme 5 --device xpu
```

Define different counts in `batch_queue.yaml`:

```yaml
jobs:
  - theme: frozen_crystal_cave
    count: 5
  - theme: vintage_winter_cabin
    count: 10
```

Run the queue:

```powershell
python generate.py --queue batch_queue.yaml --device xpu --width 6000 --height 3000
```

The short inline form is also supported:

```powershell
python generate.py --theme-counts "vintage_winter_cabin:20,frozen_crystal_cave:30,natural_coral_reef:15" --device xpu
```

Each job receives a different random seed and prompt variant, producing separate candidate SKUs for the same theme.

## Upscaling

```bash
# Honest Lanczos fallback / exact-size reframing
python upscale.py --input output/drafts --width 6000 --height 3000

# Real-ESRGAN (place the official weights in models/realesrgan)
python upscale.py --input output/drafts --width 6000 --height 3000 --use-realesrgan --model realesrnet_x4plus --tile 256 --overlap 32

# Large print
python upscale.py --input output/drafts --width 12000 --height 6000 --use-realesrgan --model realesrnet_x4plus
```

Real-ESRGAN inference is implemented with PyTorch, so the `realesrgan` and `basicsr` Python packages are not required. Download the official weights from the [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) and place `RealESRGAN_x4plus.pth` or `RealESRGAN_x4plus_anime_6B.pth` in `models/realesrgan`. Missing weights or inference errors are reported instead of silently saving a blurry Lanczos-only image when AI upscaling is enabled. The anime checkpoint is intended for anime/illustration and should be tested against painterly art.

Real-ESRGAN x4 creates plausible high-frequency detail; it cannot recover exact details that were never present in the low-resolution source. Prefer the largest stable native generation resolution, review the output at print size, and do not treat 6000×3000 dimensions alone as proof of print quality.

The upscaler uses XPU when available and tiles work to limit GPU memory; otherwise it selects CUDA, MPS, or CPU.

## Print sizing

The default master is exactly 6000×3000 PNG RGB at 150 DPI metadata. At 150 PPI, a 6000×3000 image is 40×20 inches (101.6×50.8 cm). The 12000×6000 large-print master is about 80×40 inches (203.2×101.6 cm). DPI metadata does not add pixels or guarantee print quality.

Print quality depends on native image quality, upscaling, viewing distance, and substrate. AI images require human review for composition and artifacts.
Pixel-art masters use palette quantization and nearest-neighbor enlargement instead of Real-ESRGAN to preserve the intended block-grid aesthetic. Check the final print dimensions and visible pixel scale at the intended viewing distance before listing.

## Quality and human review

QC writes status values: DRAFT, NEEDS_REVIEW, APPROVED, REJECTED, UPSCALED, or EXPORT_READY. Automated checks never certify anatomy, object correctness, or commercial suitability. Individual human review is required.

## Output

Each SKU contains `source.png`, `print_*x*.png`, `preview.jpg`, and `metadata.json`. Catalogue CSV/Excel is generated with `--export-csv` or `--export-excel`.

## Tests

```bash
pytest -q
```

Before a 100+ image run, complete `--test-one`, inspect the preview/contact sheet, and verify model/license terms, disk space, output dimensions, and print service requirements.

## Legal and listing notes

This system is a creative assistant, not a publisher. Etsy and Shopee listings must disclose AI assistance where required and must involve meaningful human prompt, editing, and curation. Do not use living artists' names, copyrighted characters, logos, landmarks, or brands. Recheck platform and model terms at listing time.
