"""Resolve Mamba imports for Windows wheels and Linux mamba2 submodule builds."""

from __future__ import annotations

import importlib.util
import inspect
import warnings
from functools import partial
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2]
_MAMBA2_SIMPLE = _REPO_SRC / "mamba2" / "mamba_ssm" / "modules" / "mamba_simple.py"


def _load_mamba2_simple_module():
    spec = importlib.util.spec_from_file_location(
        "videomamba_mamba2_simple",
        _MAMBA2_SIMPLE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bimamba_ops_available() -> bool:
    try:
        from mamba_ssm.ops.selective_scan_interface import bimamba_inner_fn

        return bimamba_inner_fn is not None
    except ImportError:
        return False


def _import_mamba():
    if _MAMBA2_SIMPLE.is_file():
        try:
            mod = _load_mamba2_simple_module()
            return mod.Mamba, mod.RMSNorm, mod.layer_norm_fn, mod.rms_norm_fn, "mamba2"
        except Exception:
            pass

    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

    return Mamba, RMSNorm, layer_norm_fn, rms_norm_fn, "wheel"


Mamba, RMSNorm, layer_norm_fn, rms_norm_fn, _MAMBA_SOURCE = _import_mamba()
_MAMBA_INIT_PARAMS = set(inspect.signature(Mamba.__init__).parameters)
_BIMAMBA_OPS_AVAILABLE = _bimamba_ops_available()


def build_mamba_mixer_cls(**kwargs):
    """Return ``partial(Mamba, ...)`` with kwargs supported by the active backend."""
    if kwargs.get("bimamba") and not _BIMAMBA_OPS_AVAILABLE:
        kwargs = dict(kwargs)
        kwargs["bimamba"] = False

    filtered = {k: v for k, v in kwargs.items() if k in _MAMBA_INIT_PARAMS}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        warnings.warn(
            f"Dropping unsupported Mamba kwargs for {_MAMBA_SOURCE} backend: "
            f"{sorted(dropped)}",
            stacklevel=2,
        )
    return partial(Mamba, **filtered)


__all__ = [
    "Mamba",
    "RMSNorm",
    "layer_norm_fn",
    "rms_norm_fn",
    "build_mamba_mixer_cls",
]
