"""Numerical hot loops used by Hawkes models.

The public API never depends on Numba-specific objects. These helpers are kept
behind small wrappers so reference NumPy/Python code remains easy to test.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised indirectly when numba is installed
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _identity_jit(func):
    return func


_jit = njit(cache=True) if njit is not None else _identity_jit


@_jit
def exp_feature_at_time(t: float, timestamps: np.ndarray, decay: float) -> float:
    value = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= t:
            break
        value += decay * np.exp(-decay * (t - tk))
    return value


@_jit
def exp_primitive_sum(end_time: float, timestamps: np.ndarray, decay: float) -> float:
    value = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        if decay == 0.0:
            continue
        value += 1.0 - np.exp(-decay * (end_time - tk))
    return value


@_jit
def sumexp_feature_at_time(
    t: float, timestamps: np.ndarray, decays: np.ndarray, out: np.ndarray
) -> None:
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= t:
            break
        delay = t - tk
        for u in range(decays.size):
            out[u] += decays[u] * np.exp(-decays[u] * delay)


@_jit
def sumexp_primitive_sum(
    end_time: float, timestamps: np.ndarray, decays: np.ndarray, out: np.ndarray
) -> None:
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        delay = end_time - tk
        for u in range(decays.size):
            out[u] += 1.0 - np.exp(-decays[u] * delay)


def finite_difference_grad(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in range(x.size):
        step = eps * max(1.0, abs(float(x[idx])))
        xp = x.copy()
        xm = x.copy()
        xp[idx] += step
        xm[idx] -= step
        grad[idx] = (func(xp) - func(xm)) / (2.0 * step)
    return grad


def finite_difference_hessian(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    for i in range(n):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        gp = finite_difference_grad(func, xp)
        gm = finite_difference_grad(func, xm)
        hess[:, i] = (gp - gm) / (2.0 * step)
    return 0.5 * (hess + hess.T)
