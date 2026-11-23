from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass


@dataclass
class HardwareInfo:
    system: str
    machine: str
    ram_gb: float
    has_mps: bool
    has_xpu: bool
    has_cuda: bool
    vram_gb: float | None
    backend: str
    torch_version: str | None


def detect_hardware() -> HardwareInfo:
    has_mps = has_xpu = has_cuda = False
    vram = None
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            vram = round(torch.cuda.mem_get_info()[1] / 1024**3, 2)
        if hasattr(torch.backends, "mps"):
            has_mps = bool(torch.backends.mps.is_available())
        if hasattr(torch, "xpu"):
            try:
                has_xpu = bool(torch.xpu.is_available())
                if hasattr(torch.xpu, "get_device_properties"):
                    vram = round(torch.xpu.get_device_properties(0).total_memory / 1024**3, 2)
            except Exception:
                has_xpu = False
    except Exception:
        pass

    ram_gb = round(_ram_bytes() / 1024**3, 2)
    system = platform.system()
    if has_mps:
        backend = "mps"
    elif has_xpu:
        backend = "xpu"
    elif has_cuda:
        backend = "cuda"
    else:
        backend = "cpu"
    return HardwareInfo(system, platform.machine(), ram_gb, has_mps, has_xpu, has_cuda, vram, backend, torch_version)


def _ram_bytes() -> int:
    try:
        if platform.system() == "Windows":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = MemoryStatus(); s.dwLength = ctypes.sizeof(s)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.ullTotalPhys
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        return 0


def recommended_backend(requested: str = "auto") -> str:
    info = detect_hardware()
    if requested != "auto":
        mapping = {"mac": "mps", "mps": "mps", "windows": "xpu", "intel": "xpu", "xpu": "xpu", "cuda": "cuda", "cpu": "cpu"}
        return mapping.get(requested.lower(), requested.lower())
    return info.backend
