$ErrorActionPreference = 'Stop'
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) { throw 'Could not install Intel XPU PyTorch wheels.' }
python -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install application dependencies.' }
Write-Host 'Use only a verified Intel XPU/OpenVINO build. Test MPS is unavailable on Windows.'
Write-Host 'Run the GUI with .\start_gui.bat, then choose the model and per-theme image counts.'
