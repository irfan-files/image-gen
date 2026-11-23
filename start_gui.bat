@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Could not create .venv. Install Python 3.12 and try again.
        pause
        exit /b 1
    )
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
"%PYTHON%" -c "import yaml" >nul 2>nul
if errorlevel 1 (
    echo Installing the missing GUI dependency into the project .venv...
    "%PYTHON%" -m pip install -r requirements-gui.txt
    if errorlevel 1 (
        echo Could not install GUI dependency. Check internet access and try again.
        pause
        exit /b 1
    )
)

"%PYTHON%" gui.py
if errorlevel 1 pause
