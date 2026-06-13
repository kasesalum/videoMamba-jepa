"""Resolve Mamba imports for Windows wheels and Linux mamba2 submodule builds."""

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    from mamba2.mamba_ssm.modules.mamba_simple import Mamba
    try:
        from mamba2.mamba_ssm.ops.triton.layer_norm import (
            RMSNorm, layer_norm_fn, rms_norm_fn,
        )
    except ImportError:
        from mamba2.mamba_ssm.ops.triton.layernorm import (
            RMSNorm, layer_norm_fn, rms_norm_fn,
        )

__all__ = ["Mamba", "RMSNorm", "layer_norm_fn", "rms_norm_fn"]
