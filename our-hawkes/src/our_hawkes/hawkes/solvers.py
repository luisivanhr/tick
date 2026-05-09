"""Small solver and proximal-operator utilities for Hawkes learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@dataclass
class ProxPositive:
    strength: float = 0.0

    def call(self, coeffs: np.ndarray, step: float = 1.0) -> np.ndarray:
        del step
        return np.maximum(np.asarray(coeffs, dtype=float), 0.0)

    def value(self, coeffs: np.ndarray) -> float:
        return 0.0 if np.all(np.asarray(coeffs) >= 0) else np.inf


@dataclass
class ProxL1:
    strength: float = 1.0

    def call(self, coeffs: np.ndarray, step: float = 1.0) -> np.ndarray:
        return soft_threshold(np.asarray(coeffs, dtype=float), self.strength * step)

    def value(self, coeffs: np.ndarray) -> float:
        return float(self.strength * np.sum(np.abs(coeffs)))


@dataclass
class ProxL2Sq:
    strength: float = 1.0

    def call(self, coeffs: np.ndarray, step: float = 1.0) -> np.ndarray:
        return np.asarray(coeffs, dtype=float) / (1.0 + self.strength * step)

    def value(self, coeffs: np.ndarray) -> float:
        return float(0.5 * self.strength * np.dot(coeffs, coeffs))


@dataclass
class ProxElasticNet:
    strength: float = 1.0
    ratio: float = 0.95

    def call(self, coeffs: np.ndarray, step: float = 1.0) -> np.ndarray:
        l1 = soft_threshold(np.asarray(coeffs, dtype=float), self.strength * self.ratio * step)
        return l1 / (1.0 + self.strength * (1.0 - self.ratio) * step)

    def value(self, coeffs: np.ndarray) -> float:
        coeffs = np.asarray(coeffs, dtype=float)
        return float(
            self.strength * self.ratio * np.sum(np.abs(coeffs))
            + 0.5 * self.strength * (1.0 - self.ratio) * np.dot(coeffs, coeffs)
        )


@dataclass
class ProxNuclear:
    strength: float = 1.0
    n_rows: int | None = None

    def call(self, coeffs: np.ndarray, step: float = 1.0) -> np.ndarray:
        matrix = self._matrix(coeffs)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        s = np.maximum(s - self.strength * step, 0.0)
        return (u * s) @ vh

    def value(self, coeffs: np.ndarray) -> float:
        matrix = self._matrix(coeffs)
        return float(self.strength * np.sum(np.linalg.svd(matrix, compute_uv=False)))

    def _matrix(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.ndim == 2:
            return coeffs
        if self.n_rows is None:
            n_rows = int(np.sqrt(coeffs.size))
        else:
            n_rows = self.n_rows
        if coeffs.size % n_rows != 0:
            raise ValueError("cannot reshape coefficients for nuclear norm")
        return coeffs.reshape((n_rows, coeffs.size // n_rows))


def group_l1_shrink(coeffs: np.ndarray, strength: float, axis: int = -1) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    norms = np.linalg.norm(coeffs, axis=axis, keepdims=True)
    scale = np.maximum(1.0 - strength / np.maximum(norms, 1e-15), 0.0)
    return coeffs * scale


def optimize_positive_coeffs(
    model,
    start: np.ndarray,
    penalty: str = "none",
    C: float = 1e3,
    elastic_net_ratio: float = 0.95,
    max_iter: int = 100,
    tol: float = 1e-5,
    jac: bool = True,
    callback: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Optimize a Hawkes model with non-negative coefficients."""

    start = np.maximum(np.asarray(start, dtype=float), 1e-12)
    penalty = penalty.lower()
    if C is None or C <= 0:
        raise ValueError("C must be positive")

    strength = 0.0 if penalty == "none" else 1.0 / C

    def objective(x: np.ndarray) -> float:
        value = model.loss(x)
        return float(value + _penalty_value(x, penalty, strength, elastic_net_ratio, model))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = model.grad(x)
        return grad + _penalty_grad(x, penalty, strength, elastic_net_ratio, model)

    use_jac = jac and penalty != "nuclear"
    result = minimize(
        objective,
        start,
        jac=gradient if use_jac else None,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * start.size,
        callback=callback,
        options={"maxiter": int(max_iter), "ftol": float(tol), "gtol": float(tol)},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"optimization failed: {result.message}")
    return np.maximum(np.asarray(result.x, dtype=float), 0.0)


def _penalty_value(x, penalty, strength, ratio, model) -> float:
    if penalty == "none":
        return 0.0
    if penalty == "l1":
        return float(strength * np.sum(np.abs(x)))
    if penalty == "l2":
        return float(0.5 * strength * np.dot(x, x))
    if penalty == "elasticnet":
        return float(strength * ratio * np.sum(np.abs(x)) + 0.5 * strength * (1.0 - ratio) * np.dot(x, x))
    if penalty == "nuclear":
        n = model.n_nodes
        matrix = x[n:].reshape((n, -1))
        return float(strength * np.sum(np.linalg.svd(matrix, compute_uv=False)))
    raise ValueError(f"unknown penalty {penalty!r}")


def _penalty_grad(x, penalty, strength, ratio, model) -> np.ndarray:
    del model
    if penalty == "none":
        return np.zeros_like(x)
    if penalty == "l1":
        return strength * np.sign(x)
    if penalty == "l2":
        return strength * x
    if penalty == "elasticnet":
        return strength * ratio * np.sign(x) + strength * (1.0 - ratio) * x
    if penalty == "nuclear":
        return np.zeros_like(x)
    raise ValueError(f"unknown penalty {penalty!r}")
