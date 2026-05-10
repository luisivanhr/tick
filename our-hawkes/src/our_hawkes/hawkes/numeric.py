"""Numerical hot loops used by Hawkes models.

Reference implementations in this module are plain Python/NumPy and remain the
source of truth. Public wrappers dispatch to Numba when available, keeping the
same signatures as the reference helpers.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - availability is environment dependent
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False


def _compile(func):
    if NUMBA_AVAILABLE:
        return njit(cache=True)(func)
    return func


def _float_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _int_array(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


def pack_realization(realization) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of node timestamp arrays into dense arrays for Numba loops."""

    arrays = [_float_array(timestamps) for timestamps in realization]
    n_nodes = len(arrays)
    sizes = np.asarray([timestamps.size for timestamps in arrays], dtype=np.int64)
    max_size = int(np.max(sizes)) if sizes.size else 0
    events = np.zeros((n_nodes, max_size), dtype=np.float64)
    for node, timestamps in enumerate(arrays):
        if timestamps.size:
            events[node, : timestamps.size] = timestamps
    return events, sizes


def exp_feature_at_time_reference(
    t: float,
    timestamps: np.ndarray,
    decay: float,
    include_current: bool = False,
) -> float:
    value = 0.0
    t = float(t)
    decay = float(decay)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk > t or (tk == t and not include_current):
            break
        value += decay * math.exp(-decay * (t - tk))
    return float(value)


def exp_primitive_sum_reference(end_time: float, timestamps: np.ndarray, decay: float) -> float:
    value = 0.0
    end_time = float(end_time)
    decay = float(decay)
    if decay == 0.0:
        return 0.0
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk >= end_time:
            break
        value += 1.0 - math.exp(-decay * (end_time - tk))
    return float(value)


def sumexp_feature_at_time_reference(
    t: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
    include_current: bool = False,
) -> None:
    out[:] = 0.0
    t = float(t)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk > t or (tk == t and not include_current):
            break
        delay = t - tk
        for u, decay in enumerate(np.asarray(decays, dtype=float)):
            out[u] += float(decay) * math.exp(-float(decay) * delay)


def sumexp_primitive_sum_reference(
    end_time: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    out[:] = 0.0
    end_time = float(end_time)
    for tk in np.asarray(timestamps, dtype=float):
        tk = float(tk)
        if tk >= end_time:
            break
        delay = end_time - tk
        for u, decay in enumerate(np.asarray(decays, dtype=float)):
            out[u] += 1.0 - math.exp(-float(decay) * delay)


def exp_kernel_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
    include_current: bool = False,
) -> float:
    return float(
        float(intensity)
        * exp_feature_at_time_reference(time, timestamps, decay, include_current)
    )


def exp_kernel_primitive_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
) -> float:
    return float(float(intensity) * exp_primitive_sum_reference(time, timestamps, decay))


def sumexp_kernel_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> float:
    features = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    sumexp_feature_at_time_reference(time, timestamps, decays, features, include_current)
    return float(np.dot(np.asarray(intensities, dtype=float), features))


def sumexp_kernel_primitive_convolution_reference(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
) -> float:
    primitives = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    sumexp_primitive_sum_reference(time, timestamps, decays, primitives)
    return float(np.dot(np.asarray(intensities, dtype=float), primitives))


def exp_intensity_vector_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    values = np.asarray(baseline, dtype=float).copy()
    n_nodes = values.size
    for i in range(n_nodes):
        for j in range(n_nodes):
            values[i] += float(adjacency[i, j]) * exp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                float(decays[i, j]),
                include_current,
            )
    return values


def sumexp_intensity_vector_reference(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    values = np.asarray(baseline, dtype=float).copy()
    n_nodes = values.size
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            sumexp_feature_at_time_reference(
                t,
                events[j, : int(sizes[j])],
                decays,
                tmp,
                include_current,
            )
            values[i] += float(np.dot(adjacency[i, j, :], tmp))
    return values


def exp_compensator_value_reference(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    value = float(baseline[node]) * float(t)
    for j in range(np.asarray(baseline).size):
        value += float(adjacency[node, j]) * exp_primitive_sum_reference(
            t, events[j, : int(sizes[j])], float(decays[node, j])
        )
    return float(value)


def sumexp_compensator_value_reference(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    value = float(baseline[node]) * float(t)
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for j in range(np.asarray(baseline).size):
        sumexp_primitive_sum_reference(t, events[j, : int(sizes[j])], decays, tmp)
        value += float(np.dot(adjacency[node, j, :], tmp))
    return float(value)


def exp_loglik_loss_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    n_nodes = np.asarray(baseline).size
    value = float((np.sum(baseline) - n_nodes) * float(end_time))
    for i in range(n_nodes):
        for j in range(n_nodes):
            value += float(adjacency[i, j]) * exp_primitive_sum_reference(
                end_time, events[j, : int(sizes[j])], float(decays[i, j])
            )
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            for j in range(n_nodes):
                intensity += float(adjacency[i, j]) * exp_feature_at_time_reference(
                    t, events[j, : int(sizes[j])], float(decays[i, j])
                )
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return float(value)


def sumexp_loglik_loss_scan_reference(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    n_nodes = np.asarray(baseline).size
    value = float((np.sum(baseline) - n_nodes) * float(end_time))
    tmp = np.empty(np.asarray(decays, dtype=float).size, dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            sumexp_primitive_sum_reference(end_time, events[j, : int(sizes[j])], decays, tmp)
            value += float(np.dot(adjacency[i, j, :], tmp))
    for i in range(n_nodes):
        for k in range(int(sizes[i])):
            t = float(events[i, k])
            intensity = float(baseline[i])
            for j in range(n_nodes):
                sumexp_feature_at_time_reference(t, events[j, : int(sizes[j])], decays, tmp)
                intensity += float(np.dot(adjacency[i, j, :], tmp))
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return float(value)


def _exp_feature_at_time_numba_impl(t, timestamps, decay, include_current):
    value = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk > t or (tk == t and not include_current):
            break
        value += decay * math.exp(-decay * (t - tk))
    return value


def _exp_primitive_sum_numba_impl(end_time, timestamps, decay):
    value = 0.0
    if decay == 0.0:
        return 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        value += 1.0 - math.exp(-decay * (end_time - tk))
    return value


def _sumexp_feature_at_time_numba_impl(t, timestamps, decays, out, include_current):
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk > t or (tk == t and not include_current):
            break
        delay = t - tk
        for u in range(decays.size):
            out[u] += decays[u] * math.exp(-decays[u] * delay)


def _sumexp_primitive_sum_numba_impl(end_time, timestamps, decays, out):
    for u in range(decays.size):
        out[u] = 0.0
    for k in range(timestamps.size):
        tk = timestamps[k]
        if tk >= end_time:
            break
        delay = end_time - tk
        for u in range(decays.size):
            out[u] += 1.0 - math.exp(-decays[u] * delay)


def _exp_intensity_vector_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current, out):
    n_nodes = baseline.size
    for i in range(n_nodes):
        value = baseline[i]
        for j in range(n_nodes):
            feature = 0.0
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk > t or (tk == t and not include_current):
                    break
                feature += decays[i, j] * math.exp(-decays[i, j] * (t - tk))
            value += adjacency[i, j] * feature
        out[i] = value


def _sumexp_intensity_vector_numba_impl(t, events, sizes, baseline, adjacency, decays, include_current, out):
    n_nodes = baseline.size
    n_decays = decays.size
    for i in range(n_nodes):
        value = baseline[i]
        for j in range(n_nodes):
            for u in range(n_decays):
                feature = 0.0
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk > t or (tk == t and not include_current):
                        break
                    feature += decays[u] * math.exp(-decays[u] * (t - tk))
                value += adjacency[i, j, u] * feature
        out[i] = value


def _exp_compensator_value_numba_impl(node, t, events, sizes, baseline, adjacency, decays):
    value = baseline[node] * t
    n_nodes = baseline.size
    for j in range(n_nodes):
        primitive = 0.0
        decay = decays[node, j]
        if decay != 0.0:
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= t:
                    break
                primitive += 1.0 - math.exp(-decay * (t - tk))
        value += adjacency[node, j] * primitive
    return value


def _sumexp_compensator_value_numba_impl(node, t, events, sizes, baseline, adjacency, decays):
    value = baseline[node] * t
    n_nodes = baseline.size
    n_decays = decays.size
    for j in range(n_nodes):
        for u in range(n_decays):
            primitive = 0.0
            decay = decays[u]
            for k in range(sizes[j]):
                tk = events[j, k]
                if tk >= t:
                    break
                primitive += 1.0 - math.exp(-decay * (t - tk))
            value += adjacency[node, j, u] * primitive
    return value


def _exp_loglik_loss_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    baseline_sum = 0.0
    for i in range(n_nodes):
        baseline_sum += baseline[i]
    value = (baseline_sum - n_nodes) * end_time

    for i in range(n_nodes):
        for j in range(n_nodes):
            primitive = 0.0
            decay = decays[i, j]
            if decay != 0.0:
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
            value += adjacency[i, j] * primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            for j in range(n_nodes):
                feature = 0.0
                decay = decays[i, j]
                for m in range(sizes[j]):
                    tk = events[j, m]
                    if tk >= t:
                        break
                    feature += decay * math.exp(-decay * (t - tk))
                intensity += adjacency[i, j] * feature
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return value


def _sumexp_loglik_loss_scan_numba_impl(events, sizes, end_time, baseline, adjacency, decays):
    n_nodes = baseline.size
    n_decays = decays.size
    baseline_sum = 0.0
    for i in range(n_nodes):
        baseline_sum += baseline[i]
    value = (baseline_sum - n_nodes) * end_time

    for i in range(n_nodes):
        for j in range(n_nodes):
            for u in range(n_decays):
                primitive = 0.0
                decay = decays[u]
                for k in range(sizes[j]):
                    tk = events[j, k]
                    if tk >= end_time:
                        break
                    primitive += 1.0 - math.exp(-decay * (end_time - tk))
                value += adjacency[i, j, u] * primitive

    for i in range(n_nodes):
        for k in range(sizes[i]):
            t = events[i, k]
            intensity = baseline[i]
            for j in range(n_nodes):
                for u in range(n_decays):
                    feature = 0.0
                    decay = decays[u]
                    for m in range(sizes[j]):
                        tk = events[j, m]
                        if tk >= t:
                            break
                        feature += decay * math.exp(-decay * (t - tk))
                    intensity += adjacency[i, j, u] * feature
            if intensity <= 0.0:
                return np.inf
            value -= math.log(intensity)
    return value


_exp_feature_at_time_numba = _compile(_exp_feature_at_time_numba_impl)
_exp_primitive_sum_numba = _compile(_exp_primitive_sum_numba_impl)
_sumexp_feature_at_time_numba = _compile(_sumexp_feature_at_time_numba_impl)
_sumexp_primitive_sum_numba = _compile(_sumexp_primitive_sum_numba_impl)
_exp_intensity_vector_numba = _compile(_exp_intensity_vector_numba_impl)
_sumexp_intensity_vector_numba = _compile(_sumexp_intensity_vector_numba_impl)
_exp_compensator_value_numba = _compile(_exp_compensator_value_numba_impl)
_sumexp_compensator_value_numba = _compile(_sumexp_compensator_value_numba_impl)
_exp_loglik_loss_scan_numba = _compile(_exp_loglik_loss_scan_numba_impl)
_sumexp_loglik_loss_scan_numba = _compile(_sumexp_loglik_loss_scan_numba_impl)


def exp_feature_at_time(t: float, timestamps: np.ndarray, decay: float) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        return float(_exp_feature_at_time_numba(float(t), timestamps, float(decay), False))
    return exp_feature_at_time_reference(t, timestamps, decay)


def exp_primitive_sum(end_time: float, timestamps: np.ndarray, decay: float) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        return float(_exp_primitive_sum_numba(float(end_time), timestamps, float(decay)))
    return exp_primitive_sum_reference(end_time, timestamps, decay)


def sumexp_feature_at_time(
    t: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    timestamps = _float_array(timestamps)
    decays = _float_array(decays)
    out_arr = np.asarray(out, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_feature_at_time_numba(float(t), timestamps, decays, out_arr, False)
    else:
        sumexp_feature_at_time_reference(t, timestamps, decays, out_arr)


def sumexp_primitive_sum(
    end_time: float,
    timestamps: np.ndarray,
    decays: np.ndarray,
    out: np.ndarray,
) -> None:
    timestamps = _float_array(timestamps)
    decays = _float_array(decays)
    out_arr = np.asarray(out, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_primitive_sum_numba(float(end_time), timestamps, decays, out_arr)
    else:
        sumexp_primitive_sum_reference(end_time, timestamps, decays, out_arr)


def exp_kernel_convolution(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
    include_current: bool = False,
) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        feature = _exp_feature_at_time_numba(
            float(time), timestamps, float(decay), bool(include_current)
        )
        return float(float(intensity) * feature)
    return exp_kernel_convolution_reference(time, timestamps, intensity, decay, include_current)


def exp_kernel_primitive_convolution(
    time: float,
    timestamps: np.ndarray,
    intensity: float,
    decay: float,
) -> float:
    timestamps = _float_array(timestamps)
    if NUMBA_AVAILABLE:
        primitive = _exp_primitive_sum_numba(float(time), timestamps, float(decay))
        return float(float(intensity) * primitive)
    return exp_kernel_primitive_convolution_reference(time, timestamps, intensity, decay)


def sumexp_kernel_convolution(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> float:
    timestamps = _float_array(timestamps)
    intensities = _float_array(intensities)
    decays = _float_array(decays)
    tmp = np.empty(decays.size, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_feature_at_time_numba(float(time), timestamps, decays, tmp, bool(include_current))
    else:
        sumexp_feature_at_time_reference(time, timestamps, decays, tmp, include_current)
    return float(np.dot(intensities, tmp))


def sumexp_kernel_primitive_convolution(
    time: float,
    timestamps: np.ndarray,
    intensities: np.ndarray,
    decays: np.ndarray,
) -> float:
    timestamps = _float_array(timestamps)
    intensities = _float_array(intensities)
    decays = _float_array(decays)
    tmp = np.empty(decays.size, dtype=np.float64)
    if NUMBA_AVAILABLE:
        _sumexp_primitive_sum_numba(float(time), timestamps, decays, tmp)
    else:
        sumexp_primitive_sum_reference(time, timestamps, decays, tmp)
    return float(np.dot(intensities, tmp))


def exp_intensity_vector(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        out = np.empty_like(baseline)
        _exp_intensity_vector_numba(
            float(t), events, sizes, baseline, adjacency, decays, bool(include_current), out
        )
        return out
    return exp_intensity_vector_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def sumexp_intensity_vector(
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    include_current: bool = False,
) -> np.ndarray:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        out = np.empty_like(baseline)
        _sumexp_intensity_vector_numba(
            float(t), events, sizes, baseline, adjacency, decays, bool(include_current), out
        )
        return out
    return sumexp_intensity_vector_reference(t, events, sizes, baseline, adjacency, decays, include_current)


def exp_compensator_value(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_exp_compensator_value_numba(int(node), float(t), events, sizes, baseline, adjacency, decays))
    return exp_compensator_value_reference(node, t, events, sizes, baseline, adjacency, decays)


def sumexp_compensator_value(
    node: int,
    t: float,
    events: np.ndarray,
    sizes: np.ndarray,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_sumexp_compensator_value_numba(int(node), float(t), events, sizes, baseline, adjacency, decays))
    return sumexp_compensator_value_reference(node, t, events, sizes, baseline, adjacency, decays)


def exp_loglik_loss_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_exp_loglik_loss_scan_numba(events, sizes, float(end_time), baseline, adjacency, decays))
    return exp_loglik_loss_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


def sumexp_loglik_loss_scan(
    events: np.ndarray,
    sizes: np.ndarray,
    end_time: float,
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
) -> float:
    events = _float_array(events)
    sizes = _int_array(sizes)
    baseline = _float_array(baseline)
    adjacency = _float_array(adjacency)
    decays = _float_array(decays)
    if NUMBA_AVAILABLE:
        return float(_sumexp_loglik_loss_scan_numba(events, sizes, float(end_time), baseline, adjacency, decays))
    return sumexp_loglik_loss_scan_reference(events, sizes, end_time, baseline, adjacency, decays)


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
