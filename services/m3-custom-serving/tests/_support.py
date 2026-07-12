"""CPU-only import helpers for the custom serving regression suite."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock


SERVING_DIR = Path(__file__).resolve().parents[1]


def load_server_module():
    """Load the production shim without importing MLX or touching a model/GPU."""
    name = "serving_test_m3_serve_batched"
    path = SERVING_DIR / "m3_serve_batched.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    real_find_spec = importlib.util.find_spec

    def offline_find_spec(package_name, *args, **kwargs):
        if package_name in {"mlx", "mlx_lm", "mlx_vlm", "m3_batch_core"}:
            return None
        return real_find_spec(package_name, *args, **kwargs)

    clean_environment = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "M3_APC": "0",
        "M3_PREFIX_CACHE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with mock.patch.dict(os.environ, clean_environment, clear=True), mock.patch(
        "importlib.util.find_spec", side_effect=offline_find_spec
    ):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_proxy_module():
    """Load the dependency-free authenticated LAN proxy from tracked source."""
    name = "serving_test_m3_lan_proxy"
    path = SERVING_DIR.parent / "m3-lan-proxy" / "m3_lan_proxy.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
