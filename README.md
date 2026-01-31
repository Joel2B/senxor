# SenXor

Python SDK and viewer for Meridian Innovation's SenXor.

Based on SenXorEvkViewer.

Hardware target: Waveshare Thermal Camera HAT (Type-C version only).
Reference: `https://www.waveshare.com/wiki/Thermal_Camera_HAT`

## Requirements

- Python 3.6+
- Dependencies in `requirements.txt`

## Install (dev)

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the viewer

```bash
python senxor.py
```

## Build Windows exe

The `build_exe.bat` script installs dependencies, runs PyInstaller, and writes
`dist\senxor.exe`.

```bat
build_exe.bat
```

Build note: only tested on Windows 8.1 (32-bit).

The build copies `settings.json` (or `settings.example.json`) to
`dist\settings.json` and creates `dist\output` if missing.

## Configuration

Edit `settings.json` for your environment. You can start from
`settings.example.json`.
