#!/usr/bin/env python3
"""
Accelerator utilities for unified device selection and logging.

Provides a single probe to select between CUDA → MPS → CPU, along with
conservative dtype and feature gates for each backend. Includes an
optional health check and a helper to log configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class AcceleratorInfo:
    backend: str                 # 'cuda' | 'mps' | 'cpu'
    device_str: str              # 'cuda' | 'mps' | 'cpu'
    can_use_4bit: bool           # BitsAndBytes quantization okay?
    recommended_dtype: str       # 'bfloat16' | 'float16' | 'float32'
    details: Dict[str, Any]


def get_accelerator(enable_mps_fallback: bool = False, do_health_check: bool = False) -> AcceleratorInfo:
    try:
        import torch
    except Exception:
        # Torch not installed; default to CPU
        return AcceleratorInfo(
            backend="cpu",
            device_str="cpu",
            can_use_4bit=False,
            recommended_dtype="float32",
            details={"torch": "not_available"},
        )

    details: Dict[str, Any] = {}

    if torch.cuda.is_available():
        device_str = "cuda"
        backend = "cuda"
        can_use_4bit = True
        recommended_dtype = "bfloat16"
        try:
            gpu_count = torch.cuda.device_count()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            details.update({"gpu_count": gpu_count, "gpu0_mem_gb": round(gpu_mem, 2)})
        except Exception:
            pass
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
        backend = "mps"
        can_use_4bit = False  # bitsandbytes not supported on MPS
        recommended_dtype = "float16"  # bf16 typically unsupported on MPS
        if enable_mps_fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            details["mps_fallback_enabled"] = True
        else:
            details["mps_fallback_enabled"] = False
    else:
        device_str = "cpu"
        backend = "cpu"
        can_use_4bit = False
        recommended_dtype = "float32"

    if do_health_check:
        try:
            import torch
            x = torch.randn((64, 64), device=device_str)
            y = torch.randn((64, 64), device=device_str)
            z = (x @ y).mean().item()
            details["health_check"] = {"ok": True, "sample_mean": float(z)}
        except Exception as e:
            details["health_check"] = {"ok": False, "error": str(e)}

    return AcceleratorInfo(
        backend=backend,
        device_str=device_str,
        can_use_4bit=can_use_4bit,
        recommended_dtype=recommended_dtype,
        details=details,
    )


def log_accelerator(logger, info: AcceleratorInfo) -> None:
    if info.backend == "cuda":
        gc = info.details.get("gpu_count", "?")
        gm = info.details.get("gpu0_mem_gb", "?")
        logger.info(f"🎮 CUDA available: {gc} GPU(s), {gm}GB on GPU0")
    elif info.backend == "mps":
        fallback = info.details.get("mps_fallback_enabled", False)
        msg = "🍏 Apple Metal (MPS) available"
        if fallback:
            msg += " with fallback enabled"
        logger.info(msg)
    else:
        logger.warning("⚠️ No GPU/MPS accelerator detected - using CPU")

