"""
Compute Backend
===============

Thin abstraction over numpy / JAX.  When JAX is available and enabled,
array operations run on TPU/GPU via jax.numpy.  Otherwise falls back to
plain numpy with zero overhead.

Usage::

    from .backend import xp, svd, use_jax

    # xp is either numpy or jax.numpy
    theta = xp.tensordot(A, B, axes=([2], [0]))
    U, S, Vh = svd(mat)
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# State: which backend is active
# ---------------------------------------------------------------------------

_USE_JAX = False
xp = np  # array module: numpy or jax.numpy


def use_jax(enable: bool = True) -> bool:
    """Enable or disable JAX backend.  Returns True if JAX is now active."""
    global _USE_JAX, xp
    if enable:
        try:
            import jax
            import jax.numpy as jnp
            # Ensure JAX sees the TPU/GPU
            devices = jax.devices()
            xp = jnp
            _USE_JAX = True
            print(f"JAX backend enabled — devices: {[str(d) for d in devices]}")
            return True
        except ImportError:
            print("JAX not installed — using numpy. Install with: pip install jax[tpu]")
            xp = np
            _USE_JAX = False
            return False
    else:
        xp = np
        _USE_JAX = False
        return False


def is_jax() -> bool:
    return _USE_JAX


# ---------------------------------------------------------------------------
# SVD wrapper — JAX and numpy have slightly different APIs
# ---------------------------------------------------------------------------

def svd(matrix, full_matrices: bool = False):
    """SVD that works on both numpy and JAX arrays.

    Falls back to scipy's more robust 'gesvd' driver if numpy's
    default LAPACK 'gesdd' fails to converge.

    Returns (U, S, Vh) — same convention as numpy.
    """
    if _USE_JAX:
        import jax.numpy as jnp
        return jnp.linalg.svd(matrix, full_matrices=full_matrices)

    try:
        return np.linalg.svd(matrix, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        # gesdd failed — retry with the slower but more robust gesvd
        try:
            from scipy.linalg import svd as scipy_svd
            return scipy_svd(matrix, full_matrices=full_matrices,
                             lapack_driver="gesvd")
        except ImportError:
            # No scipy — clamp tiny values and retry
            matrix = np.where(np.abs(matrix) < 1e-15, 0.0, matrix)
            return np.linalg.svd(matrix, full_matrices=full_matrices)


def to_numpy(arr) -> np.ndarray:
    """Convert any array (JAX or numpy) to a plain numpy array."""
    if _USE_JAX:
        import jax
        if isinstance(arr, jax.Array):
            return np.asarray(arr)
    return np.asarray(arr)


def from_numpy(arr):
    """Convert a numpy array to the active backend's array type."""
    if _USE_JAX:
        import jax.numpy as jnp
        return jnp.asarray(arr)
    return arr
