"""Hawkes model losses and derivatives."""

from __future__ import annotations

import math
import warnings
from itertools import product
from typing import Any

import numpy as np
from scipy.integrate import quad

from our_hawkes.base import BaseEstimator, normalize_events

from .numeric import (
    exp_feature_at_time,
    exp_loglik_loss_scan,
    exp_primitive_sum,
    finite_difference_grad,
    finite_difference_hessian,
    pack_realization,
    sumexp_feature_at_time,
    sumexp_loglik_loss_scan,
    sumexp_primitive_sum,
)


class ModelHawkes(BaseEstimator):
    """Base class for Hawkes models."""

    def __init__(self, approx: int = 0, n_threads: int = 1):
        self.approx = approx
        self.n_threads = n_threads
        self.data: list[list[np.ndarray]] | None = None
        self._end_times: np.ndarray | None = None
        self._fitted = False
        self.dtype = np.dtype("float64")

    def fit(self, data: list[Any], end_times: Any | None = None):
        self.data, self._end_times, self._n_nodes = normalize_events(data, end_times)
        self._fitted = True
        self._after_set_data()
        return self

    def incremental_fit(self, events: list[Any], end_time: float | None = None):
        realizations, end_times, n_nodes = normalize_events(events, end_time)
        if len(realizations) != 1:
            raise ValueError("incremental_fit accepts a single realization")
        if self.data is None:
            self.data = []
            self._end_times = np.asarray([], dtype=float)
            self._n_nodes = n_nodes
        elif n_nodes != self.n_nodes:
            raise ValueError("incremental realization has wrong number of nodes")
        self.data.append(realizations[0])
        self._end_times = np.append(self._end_times, end_times[0])
        self._fitted = True
        self._after_set_data()
        return self

    def _after_set_data(self) -> None:
        pass

    @property
    def n_nodes(self) -> int:
        if not self._fitted:
            raise ValueError("call fit before accessing n_nodes")
        return int(self._n_nodes)

    @property
    def end_times(self) -> np.ndarray | None:
        return None if self._end_times is None else self._end_times.copy()

    @end_times.setter
    def end_times(self, val):
        if self._fitted:
            raise RuntimeError("cannot set end_times once fitted")
        self._end_times = None if val is None else np.asarray(val, dtype=float)

    @property
    def n_jumps(self) -> int:
        if self.data is None:
            return 0
        return int(sum(ts.size for realization in self.data for ts in realization))

    @property
    def n_coeffs(self) -> int:
        return self._get_n_coeffs()

    def _get_n_coeffs(self) -> int:
        raise NotImplementedError

    def loss(self, coeffs: np.ndarray) -> float:
        self._check_ready()
        return float(self._loss(np.asarray(coeffs, dtype=float)))

    def grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        self._check_ready()
        coeffs = np.asarray(coeffs, dtype=float)
        values = self._grad(coeffs)
        if out is not None:
            out[:] = values
            return out
        return values

    def loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray | None = None):
        value = self.loss(coeffs)
        grad = self.grad(coeffs, out)
        return value, grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        self._check_ready()
        return finite_difference_hessian(self.loss, np.asarray(x, dtype=float))

    def hessian_norm(self, coeffs: np.ndarray, vector: np.ndarray) -> float:
        hessian = self.hessian(coeffs)
        vector = np.asarray(vector, dtype=float)
        value = float(vector @ hessian @ vector)
        return value

    def _loss(self, coeffs: np.ndarray) -> float:
        raise NotImplementedError

    def _grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        values = finite_difference_grad(self.loss, coeffs)
        if out is not None:
            out[:] = values
            return out
        return values

    def _loss_and_grad(self, coeffs: np.ndarray, out: np.ndarray | None = None):
        value = self._loss(np.asarray(coeffs, dtype=float))
        grad = self._grad(np.asarray(coeffs, dtype=float), out)
        return value, grad

    def _check_ready(self) -> None:
        if not self._fitted or self.data is None or self._end_times is None:
            raise ValueError("call fit before using the model")


class ModelHawkesExpKernLogLik(ModelHawkes):
    """Negative log-likelihood for fixed-decay exponential Hawkes kernels."""

    def __init__(self, decay: float, n_threads: int = 1):
        super().__init__(approx=0, n_threads=n_threads)
        if decay <= 0:
            raise ValueError("decay must be positive")
        self.decay = float(decay)
        self.decays = self.decay

    def _after_set_data(self) -> None:
        self.decays_matrix = self._current_decays_matrix()

    def _current_decays_matrix(self) -> np.ndarray:
        return np.full((self.n_nodes, self.n_nodes), float(self.decay), dtype=float)

    def _get_n_coeffs(self) -> int:
        return self.n_nodes + self.n_nodes * self.n_nodes

    def _unpack(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coeffs.shape != (self.n_coeffs,):
            raise ValueError(f"coeffs should have shape {(self.n_coeffs,)}")
        baseline = coeffs[: self.n_nodes]
        adjacency = coeffs[self.n_nodes :].reshape((self.n_nodes, self.n_nodes))
        return baseline, adjacency

    def _loss(self, coeffs: np.ndarray) -> float:
        baseline, adjacency = self._unpack(coeffs)
        if np.any(baseline < 0) or np.any(adjacency < 0):
            return np.inf
        decays = self._current_decays_matrix()
        total = 0.0
        for realization, end_time in zip(self.data, self._end_times):
            total += _exp_loglik_loss_one(realization, float(end_time), baseline, adjacency, decays)
        return total / max(self.n_jumps, 1)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        baseline, adjacency = self._unpack(coeffs)
        decays = self._current_decays_matrix()
        grad_baseline = np.zeros_like(baseline)
        grad_adjacency = np.zeros_like(adjacency)
        for realization, end_time in zip(self.data, self._end_times):
            gb, ga = _exp_loglik_grad_one(realization, float(end_time), baseline, adjacency, decays)
            grad_baseline += gb
            grad_adjacency += ga
        grad = np.hstack((grad_baseline, grad_adjacency.ravel()))
        grad = grad / max(self.n_jumps, 1)
        if out is not None:
            out[:] = grad
            return out
        return grad


class ModelHawkesSumExpKernLogLik(ModelHawkes):
    """Negative log-likelihood for fixed sum-exponential Hawkes kernels."""

    def __init__(self, decays: Any, n_threads: int = 1):
        super().__init__(approx=0, n_threads=n_threads)
        self.decays = np.asarray(decays, dtype=float)
        if self.decays.ndim != 1 or self.decays.size == 0:
            raise ValueError("decays must be a non-empty one-dimensional array")
        if np.any(self.decays <= 0):
            raise ValueError("decays must be positive")

    @property
    def n_decays(self) -> int:
        return int(self._current_decays().size)

    def _current_decays(self) -> np.ndarray:
        decays = np.asarray(self.decays, dtype=float)
        if decays.ndim != 1 or decays.size == 0:
            raise ValueError("decays must be a non-empty one-dimensional array")
        if np.any(decays <= 0):
            raise ValueError("decays must be positive")
        return decays

    def _get_n_coeffs(self) -> int:
        return self.n_nodes + self.n_nodes * self.n_nodes * self.n_decays

    def _unpack(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coeffs.shape != (self.n_coeffs,):
            raise ValueError(f"coeffs should have shape {(self.n_coeffs,)}")
        baseline = coeffs[: self.n_nodes]
        adjacency = coeffs[self.n_nodes :].reshape((self.n_nodes, self.n_nodes, self.n_decays))
        return baseline, adjacency

    def _loss(self, coeffs: np.ndarray) -> float:
        baseline, adjacency = self._unpack(coeffs)
        if np.any(baseline < 0) or np.any(adjacency < 0):
            return np.inf
        decays = self._current_decays()
        total = 0.0
        for realization, end_time in zip(self.data, self._end_times):
            total += _sumexp_loglik_loss_one(realization, float(end_time), baseline, adjacency, decays)
        return total / max(self.n_jumps, 1)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        baseline, adjacency = self._unpack(coeffs)
        decays = self._current_decays()
        grad_baseline = np.zeros_like(baseline)
        grad_adjacency = np.zeros_like(adjacency)
        for realization, end_time in zip(self.data, self._end_times):
            gb, ga = _sumexp_loglik_grad_one(realization, float(end_time), baseline, adjacency, decays)
            grad_baseline += gb
            grad_adjacency += ga
        grad = np.hstack((grad_baseline, grad_adjacency.ravel()))
        grad = grad / max(self.n_jumps, 1)
        if out is not None:
            out[:] = grad
            return out
        return grad


class ModelHawkesExpKernLeastSq(ModelHawkes):
    """Grid-based least-squares loss for exponential Hawkes kernels."""

    def __init__(self, decays: float | Any, approx: int = 0, n_threads: int = 1):
        super().__init__(approx=approx, n_threads=n_threads)
        self.decays = decays

    def _after_set_data(self) -> None:
        self.decays_matrix = self._current_decays_matrix()

    def _current_decays_matrix(self) -> np.ndarray:
        if isinstance(self.decays, (int, float, np.floating)):
            decays_matrix = np.full((self.n_nodes, self.n_nodes), float(self.decays))
        else:
            arr = np.asarray(self.decays, dtype=float)
            if arr.shape != (self.n_nodes, self.n_nodes):
                raise ValueError("decays must be scalar or an n_nodes by n_nodes array")
            decays_matrix = arr
        if np.any(decays_matrix <= 0):
            raise ValueError("decays must be positive")
        return decays_matrix

    def _get_n_coeffs(self) -> int:
        return self.n_nodes + self.n_nodes * self.n_nodes

    def _unpack(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coeffs.shape != (self.n_coeffs,):
            raise ValueError(f"coeffs should have shape {(self.n_coeffs,)}")
        baseline = coeffs[: self.n_nodes]
        adjacency = coeffs[self.n_nodes :].reshape((self.n_nodes, self.n_nodes))
        return baseline, adjacency

    def _loss(self, coeffs: np.ndarray) -> float:
        baseline, adjacency = self._unpack(coeffs)
        if np.any(baseline < 0) or np.any(adjacency < 0):
            return np.inf
        decays = self._current_decays_matrix()
        total = 0.0
        for realization, end_time in zip(self.data, self._end_times):
            total += _exp_least_squares_one(realization, float(end_time), baseline, adjacency, decays)
        return total / max(self.n_jumps, 1)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        baseline, adjacency = self._unpack(coeffs)
        decays = self._current_decays_matrix()
        grad_baseline = np.zeros_like(baseline)
        grad_adjacency = np.zeros_like(adjacency)
        for realization, end_time in zip(self.data, self._end_times):
            gb, ga = _exp_least_squares_grad_one(
                realization, float(end_time), baseline, adjacency, decays
            )
            grad_baseline += gb
            grad_adjacency += ga
        grad = np.hstack((grad_baseline, grad_adjacency.ravel()))
        grad = grad / max(self.n_jumps, 1)
        if out is not None:
            out[:] = grad
            return out
        return grad


class ModelHawkesSumExpKernLeastSq(ModelHawkes):
    """Grid-based least-squares loss for sum-exponential Hawkes kernels."""

    def __init__(
        self,
        decays: Any,
        n_baselines: int = 1,
        period_length: float | None = None,
        approx: int = 0,
        n_threads: int = 1,
    ):
        super().__init__(approx=approx, n_threads=n_threads)
        self.decays = np.asarray(decays, dtype=float)
        if self.decays.ndim != 1 or self.decays.size == 0:
            raise ValueError("decays must be a non-empty one-dimensional array")
        if np.any(self.decays <= 0):
            raise ValueError("decays must be positive")
        if n_baselines <= 0:
            raise ValueError("n_baselines must be positive")
        if n_baselines > 1 and period_length is None:
            raise ValueError("period_length must be given if multiple baselines are used")
        if n_baselines == 1 and period_length is not None:
            warnings.warn(
                "period_length has no effect when using a constant baseline",
                UserWarning,
                stacklevel=2,
            )
        self.n_baselines = int(n_baselines)
        self.period_length = period_length

    @property
    def n_decays(self) -> int:
        return int(self._current_decays().size)

    def _current_decays(self) -> np.ndarray:
        decays = np.asarray(self.decays, dtype=float)
        if decays.ndim != 1 or decays.size == 0:
            raise ValueError("decays must be a non-empty one-dimensional array")
        if np.any(decays <= 0):
            raise ValueError("decays must be positive")
        return decays

    @property
    def baseline_intervals(self) -> np.ndarray:
        if self.n_baselines == 1:
            return np.asarray([0.0])
        return np.arange(self.n_baselines, dtype=float) * (self.period_length / self.n_baselines)

    def cast_period_length(self, period_length: float | None = None) -> float:
        """Return tick's C++ sentinel for an unset varying-baseline period."""

        if period_length is None:
            period_length = self.period_length
        if period_length is None:
            return float(np.finfo(np.float64).max)
        return float(period_length)

    def _get_n_coeffs(self) -> int:
        return self.n_nodes * self.n_baselines + self.n_nodes * self.n_nodes * self.n_decays

    def _unpack(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coeffs.shape != (self.n_coeffs,):
            raise ValueError(f"coeffs should have shape {(self.n_coeffs,)}")
        baseline_raw = coeffs[: self.n_nodes * self.n_baselines]
        if self.n_baselines == 1:
            baseline = baseline_raw
        else:
            baseline = baseline_raw.reshape((self.n_nodes, self.n_baselines))
        adjacency = coeffs[self.n_nodes * self.n_baselines :].reshape(
            (self.n_nodes, self.n_nodes, self.n_decays)
        )
        return baseline, adjacency

    def _loss(self, coeffs: np.ndarray) -> float:
        baseline, adjacency = self._unpack(coeffs)
        if np.any(baseline < 0) or np.any(adjacency < 0):
            return np.inf
        decays = self._current_decays()
        total = 0.0
        for realization, end_time in zip(self.data, self._end_times):
            total += _sumexp_least_squares_one(
                realization,
                float(end_time),
                baseline,
                adjacency,
                decays,
                self.period_length,
            )
        return total / max(self.n_jumps, 1)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        baseline, adjacency = self._unpack(coeffs)
        decays = self._current_decays()
        grad_baseline = np.zeros_like(np.asarray(baseline, dtype=float))
        grad_adjacency = np.zeros_like(adjacency)
        for realization, end_time in zip(self.data, self._end_times):
            gb, ga = _sumexp_least_squares_grad_one(
                realization,
                float(end_time),
                baseline,
                adjacency,
                decays,
                self.period_length,
            )
            grad_baseline += gb
            grad_adjacency += ga
        grad = np.hstack((grad_baseline.ravel(), grad_adjacency.ravel()))
        grad = grad / max(self.n_jumps, 1)
        if out is not None:
            out[:] = grad
            return out
        return grad


def _exp_loglik_loss_one(realization, end_time, baseline, adjacency, decays):
    events, sizes = pack_realization(realization)
    return exp_loglik_loss_scan(events, sizes, end_time, baseline, adjacency, decays)


def _exp_loglik_grad_one(realization, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    grad_baseline = np.full(n_nodes, end_time, dtype=float)
    grad_adjacency = np.empty((n_nodes, n_nodes), dtype=float)
    for i, j in product(range(n_nodes), range(n_nodes)):
        grad_adjacency[i, j] = exp_primitive_sum(end_time, realization[j], decays[i, j])
    for i in range(n_nodes):
        for t in realization[i]:
            features = np.empty(n_nodes, dtype=float)
            intensity = baseline[i]
            for j in range(n_nodes):
                features[j] = exp_feature_at_time(float(t), realization[j], decays[i, j])
                intensity += adjacency[i, j] * features[j]
            if intensity <= 0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            grad_adjacency[i, :] -= features / intensity
    return grad_baseline, grad_adjacency


def _sumexp_loglik_loss_one(realization, end_time, baseline, adjacency, decays):
    events, sizes = pack_realization(realization)
    return sumexp_loglik_loss_scan(events, sizes, end_time, baseline, adjacency, decays)


def _sumexp_loglik_grad_one(realization, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    n_decays = decays.size
    grad_baseline = np.full(n_nodes, end_time, dtype=float)
    grad_adjacency = np.empty_like(adjacency)
    tmp = np.empty(n_decays, dtype=float)
    for i, j in product(range(n_nodes), range(n_nodes)):
        sumexp_primitive_sum(end_time, realization[j], decays, tmp)
        grad_adjacency[i, j, :] = tmp
    features = np.empty(n_decays, dtype=float)
    event_features = np.empty((n_nodes, n_decays), dtype=float)
    for i in range(n_nodes):
        for t in realization[i]:
            intensity = baseline[i]
            for j in range(n_nodes):
                sumexp_feature_at_time(float(t), realization[j], decays, features)
                event_features[j, :] = features
                intensity += float(np.dot(adjacency[i, j, :], features))
            if intensity <= 0:
                continue
            grad_baseline[i] -= 1.0 / intensity
            grad_adjacency[i, :, :] -= event_features / intensity
    return grad_baseline, grad_adjacency


def _exp_least_squares_one(realization, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    value = 0.0
    for i in range(n_nodes):
        feature_integrals, feature_products, event_feature_sums = _exp_ls_statistics(
            realization, float(end_time), decays[i], target_node=i
        )
        coefficients = adjacency[i]
        value += float(baseline[i] ** 2 * end_time)
        value += 2.0 * float(baseline[i] * np.dot(coefficients, feature_integrals))
        value += float(coefficients @ feature_products @ coefficients)
        value -= 2.0 * float(
            baseline[i] * realization[i].size + np.dot(coefficients, event_feature_sums)
        )
    return float(value)


def _exp_least_squares_grad_one(realization, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    grad_baseline = np.zeros(n_nodes, dtype=float)
    grad_adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        feature_integrals, feature_products, event_feature_sums = _exp_ls_statistics(
            realization, float(end_time), decays[i], target_node=i
        )
        coefficients = adjacency[i]
        sym_products = 0.5 * (feature_products + feature_products.T)
        grad_baseline[i] = (
            2.0 * baseline[i] * end_time
            + 2.0 * float(np.dot(coefficients, feature_integrals))
            - 2.0 * realization[i].size
        )
        grad_adjacency[i, :] = (
            2.0 * baseline[i] * feature_integrals
            + 2.0 * sym_products @ coefficients
            - 2.0 * event_feature_sums
        )
    return grad_baseline, grad_adjacency


def _exp_ls_statistics(realization, end_time, target_decays, target_node):
    n_nodes = len(realization)
    feature_integrals = np.empty(n_nodes, dtype=float)
    event_feature_sums = np.zeros(n_nodes, dtype=float)
    feature_products = np.empty((n_nodes, n_nodes), dtype=float)

    for j in range(n_nodes):
        feature_integrals[j] = exp_primitive_sum(end_time, realization[j], target_decays[j])
        for event_time in realization[target_node]:
            event_feature_sums[j] += exp_feature_at_time(
                float(event_time), realization[j], target_decays[j]
            )

    for j, k in product(range(n_nodes), range(n_nodes)):
        feature_products[j, k] = _feature_product_integral(
            end_time,
            realization[j],
            float(target_decays[j]),
            realization[k],
            float(target_decays[k]),
        )
    return feature_integrals, feature_products, event_feature_sums


def _sumexp_least_squares_one(realization, end_time, baseline, adjacency, decays, period_length):
    if np.asarray(baseline, dtype=float).ndim != 1:
        return _sumexp_least_squares_one_quad(
            realization, end_time, baseline, adjacency, decays, period_length
        )
    n_nodes = adjacency.shape[0]
    n_decays = decays.size
    feature_integrals, feature_products = _sumexp_ls_integral_statistics(
        realization, float(end_time), decays
    )
    flat_integrals = feature_integrals.ravel()
    value = 0.0
    for i in range(n_nodes):
        coefficients = adjacency[i].reshape(n_nodes * n_decays)
        event_feature_sums = _sumexp_ls_event_feature_sums(realization, decays, i).ravel()
        value += float(baseline[i] ** 2 * end_time)
        value += 2.0 * float(baseline[i] * np.dot(coefficients, flat_integrals))
        value += float(coefficients @ feature_products @ coefficients)
        value -= 2.0 * float(
            baseline[i] * realization[i].size + np.dot(coefficients, event_feature_sums)
        )
    return float(value)


def _sumexp_least_squares_grad_one(realization, end_time, baseline, adjacency, decays, period_length):
    if np.asarray(baseline, dtype=float).ndim != 1:
        return _sumexp_least_squares_grad_one_quad(
            realization, end_time, baseline, adjacency, decays, period_length
        )
    n_nodes = adjacency.shape[0]
    n_decays = decays.size
    grad_baseline = np.zeros(n_nodes, dtype=float)
    grad_adjacency = np.zeros_like(adjacency)
    feature_integrals, feature_products = _sumexp_ls_integral_statistics(
        realization, float(end_time), decays
    )
    flat_integrals = feature_integrals.ravel()
    sym_products = 0.5 * (feature_products + feature_products.T)
    for i in range(n_nodes):
        coefficients = adjacency[i].reshape(n_nodes * n_decays)
        event_feature_sums = _sumexp_ls_event_feature_sums(realization, decays, i).ravel()
        grad_baseline[i] = (
            2.0 * baseline[i] * end_time
            + 2.0 * float(np.dot(coefficients, flat_integrals))
            - 2.0 * realization[i].size
        )
        grad_adjacency[i, :, :] = (
            2.0 * baseline[i] * flat_integrals
            + 2.0 * sym_products @ coefficients
            - 2.0 * event_feature_sums
        ).reshape(n_nodes, n_decays)
    return grad_baseline, grad_adjacency


def _sumexp_ls_integral_statistics(realization, end_time, decays):
    n_nodes = len(realization)
    n_decays = decays.size
    feature_integrals = np.empty((n_nodes, n_decays), dtype=float)
    tmp = np.empty(n_decays, dtype=float)
    for j in range(n_nodes):
        sumexp_primitive_sum(end_time, realization[j], decays, tmp)
        feature_integrals[j, :] = tmp

    n_features = n_nodes * n_decays
    feature_products = np.empty((n_features, n_features), dtype=float)
    for j, u, k, v in product(range(n_nodes), range(n_decays), range(n_nodes), range(n_decays)):
        left = j * n_decays + u
        right = k * n_decays + v
        feature_products[left, right] = _feature_product_integral(
            end_time,
            realization[j],
            float(decays[u]),
            realization[k],
            float(decays[v]),
        )
    return feature_integrals, feature_products


def _sumexp_ls_event_feature_sums(realization, decays, target_node):
    n_nodes = len(realization)
    n_decays = decays.size
    event_feature_sums = np.zeros((n_nodes, n_decays), dtype=float)
    features = np.empty(n_decays, dtype=float)
    for event_time in realization[target_node]:
        for j in range(n_nodes):
            sumexp_feature_at_time(float(event_time), realization[j], decays, features)
            event_feature_sums[j, :] += features
    return event_feature_sums


def _feature_product_integral(end_time, timestamps_a, decay_a, timestamps_b, decay_b):
    if decay_a <= 0.0 or decay_b <= 0.0:
        return 0.0
    rate = decay_a + decay_b
    value = 0.0
    for ta in timestamps_a:
        ta = float(ta)
        if ta >= end_time:
            break
        for tb in timestamps_b:
            tb = float(tb)
            if tb >= end_time:
                break
            start = max(ta, tb)
            remaining = end_time - start
            if remaining <= 0.0:
                continue
            value += (
                decay_a
                * decay_b
                * math.exp(-decay_a * (start - ta) - decay_b * (start - tb))
                * (-math.expm1(-rate * remaining))
                / rate
            )
    return float(value)


def _sumexp_least_squares_one_quad(realization, end_time, baseline, adjacency, decays, period_length):
    n_nodes = adjacency.shape[0]
    points = _quad_points(realization, end_time, period_length, np.asarray(baseline).shape[1])
    value = 0.0
    for i in range(n_nodes):
        integral = quad(
            lambda t, i=i: _sumexp_intensity_from_coeffs(
                realization, baseline, adjacency, decays, i, float(t), period_length
            )
            ** 2,
            0.0,
            end_time,
            points=points,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=1000,
        )[0]
        value += integral
        value -= 2.0 * sum(
            _sumexp_intensity_from_coeffs(
                realization, baseline, adjacency, decays, i, float(t), period_length
            )
            for t in realization[i]
        )
    return float(value)


def _sumexp_least_squares_grad_one_quad(realization, end_time, baseline, adjacency, decays, period_length):
    baseline_arr = np.asarray(baseline, dtype=float)
    n_nodes, n_baselines = baseline_arr.shape
    n_decays = decays.size
    grad_baseline = np.zeros_like(baseline_arr)
    grad_adjacency = np.zeros_like(adjacency)
    points = _quad_points(realization, end_time, period_length, n_baselines)

    for i in range(n_nodes):
        for b in range(n_baselines):
            integral = quad(
                lambda t, i=i, b=b: _sumexp_intensity_from_coeffs(
                    realization, baseline, adjacency, decays, i, float(t), period_length
                )
                * _baseline_indicator(b, float(t), period_length, n_baselines),
                0.0,
                end_time,
                points=points,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=1000,
            )[0]
            event_sum = sum(
                _baseline_indicator(b, float(t), period_length, n_baselines)
                for t in realization[i]
            )
            grad_baseline[i, b] = 2.0 * integral - 2.0 * event_sum

        for j in range(n_nodes):
            for u in range(n_decays):
                integral = quad(
                    lambda t, i=i, j=j, u=u: _sumexp_intensity_from_coeffs(
                        realization, baseline, adjacency, decays, i, float(t), period_length
                    )
                    * exp_feature_at_time(float(t), realization[j], float(decays[u])),
                    0.0,
                    end_time,
                    points=points,
                    epsabs=1e-10,
                    epsrel=1e-10,
                    limit=1000,
                )[0]
                event_sum = sum(
                    exp_feature_at_time(float(t), realization[j], float(decays[u]))
                    for t in realization[i]
                )
                grad_adjacency[i, j, u] = 2.0 * integral - 2.0 * event_sum
    return grad_baseline, grad_adjacency


def _sumexp_intensity_from_coeffs(realization, baseline, adjacency, decays, i, t, period_length):
    value = _baseline_value_from_coeffs(baseline, i, t, period_length)
    features = np.empty(decays.size, dtype=float)
    for j in range(adjacency.shape[1]):
        sumexp_feature_at_time(t, realization[j], decays, features)
        value += float(np.dot(adjacency[i, j, :], features))
    return float(value)


def _baseline_indicator(interval, t, period_length, n_intervals):
    idx = _baseline_interval_index(t, period_length, n_intervals)
    return 1.0 if idx == interval else 0.0


def _baseline_value_from_coeffs(baseline, i, t, period_length):
    arr = np.asarray(baseline, dtype=float)
    if arr.ndim == 1:
        return float(arr[i])
    if period_length is None:
        raise ValueError("period_length is required for varying baselines")
    idx = _baseline_interval_index(t, period_length, arr.shape[1])
    return float(arr[i, idx])


def _baseline_interval_index(t, period_length, n_intervals):
    idx = int(np.floor(((t % period_length) / period_length) * n_intervals))
    return min(idx, n_intervals - 1)


def _quad_points(realization, end_time, period_length, n_intervals):
    event_times = np.concatenate(
        [ts for ts in realization if ts.size] or [np.asarray([], dtype=float)]
    )
    points = [float(t) for t in event_times if 0.0 < t < end_time]
    if period_length is not None and n_intervals > 1:
        step = period_length / n_intervals
        if step > 0:
            k = 1
            while k * step < end_time:
                points.append(float(k * step))
                k += 1
    if not points:
        return None
    return sorted(set(points))
