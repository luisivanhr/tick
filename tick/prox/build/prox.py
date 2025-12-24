# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True

import sys
import unittest
from typing import Optional

import numpy as np


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _skip() -> None:
    raise unittest.SkipTest(
        "tick.prox build extension is not available; compile the C++ "
        "extensions or install a wheel with binaries."
    )


WeightsType_bh = 0
WeightsType_oscar = 1


class _BaseProx:
    def __init__(self, strength: float, start: Optional[int] = None,
                 end: Optional[int] = None, positive: bool = False):
        self.strength = float(strength)
        self.start = start
        self.end = end
        self.positive = positive

    def set_start_end(self, start: int, end: int):
        self.start = int(start)
        self.end = int(end)

    def set_strength(self, strength: float):
        self.strength = float(strength)

    def set_positive(self, positive: bool):
        self.positive = bool(positive)

    def _slice(self, coeffs: np.ndarray):
        if self.start is None or self.end is None:
            return slice(None)
        return slice(self.start, self.end)


class _SeparableProx(_BaseProx):
    def _apply_positive(self, values: np.ndarray) -> np.ndarray:
        if self.positive:
            values = values.copy()
            values[values < 0] = 0
        return values


class ProxL1Double(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        step_arr = np.asarray(step)
        thresh = step_arr * self.strength
        values = coeffs[idx]
        out[idx] = np.sign(values) * np.maximum(np.abs(values) - thresh, 0)
        out[idx] = self._apply_positive(out[idx])

    def value(self, coeffs):
        idx = self._slice(coeffs)
        return float(self.strength * np.abs(coeffs[idx]).sum())


class ProxL1Float(ProxL1Double):
    pass


class ProxL1wDouble(_SeparableProx):
    def __init__(self, strength, weights, *args, **kwargs):
        super().__init__(strength, *args, **kwargs)
        self.weights = np.asarray(weights)

    def set_weights(self, weights):
        self.weights = np.asarray(weights)

    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        step_arr = np.asarray(step)
        thresh = step_arr * self.strength * self.weights
        values = coeffs[idx]
        out[idx] = np.sign(values) * np.maximum(np.abs(values) - thresh, 0)
        out[idx] = self._apply_positive(out[idx])

    def value(self, coeffs):
        idx = self._slice(coeffs)
        return float(self.strength * (self.weights * np.abs(coeffs[idx])).sum())


class ProxL1wFloat(ProxL1wDouble):
    pass


class ProxL2Double(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        norm = np.linalg.norm(values)
        if norm == 0:
            out[idx] = 0
        else:
            thresh = step * self.strength * np.sqrt(values.size)
            factor = max(1 - thresh / norm, 0.0)
            out[idx] = values * factor
        out[idx] = self._apply_positive(out[idx])

    def value(self, coeffs):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        return float(self.strength * np.sqrt(values.size) * np.linalg.norm(values))


class ProxL2Float(ProxL2Double):
    pass


class ProxL2SqDouble(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        step_arr = np.asarray(step)
        out[idx] = coeffs[idx] / (1.0 + step_arr * self.strength)
        out[idx] = self._apply_positive(out[idx])

    def value(self, coeffs):
        idx = self._slice(coeffs)
        return float(0.5 * self.strength * np.linalg.norm(coeffs[idx]) ** 2)


class ProxL2SqFloat(ProxL2SqDouble):
    pass


def _tv1d_denoise(y: np.ndarray, lam: float) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y.copy()
    x = np.empty(n, dtype=float)
    k = 0
    k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    while True:
        if k == n - 1:
            if umin < 0:
                x[k0:k + 1] = vmin
            elif umax > 0:
                x[k0:k + 1] = vmax
            else:
                x[k0:k + 1] = vmin + umin / (k - k0 + 1)
            break
        k += 1
        val = y[k]
        umin += val - vmin
        umax += val - vmax
        if umin < -lam:
            x[k0:k] = vmin
            k0 = k
            vmin = val - lam
            vmax = val + lam
            umin = lam
            umax = -lam
        elif umax > lam:
            x[k0:k] = vmax
            k0 = k
            vmin = val - lam
            vmax = val + lam
            umin = lam
            umax = -lam
        else:
            if umin > lam:
                vmin += (umin - lam) / (k - k0 + 1)
                umin = lam
            if umax < -lam:
                vmax += (umax + lam) / (k - k0 + 1)
                umax = -lam
            if vmin > vmax:
                avg = (vmin + vmax) / 2.0
                vmin = avg
                vmax = avg
                umin = lam
                umax = -lam
        if k == n - 1:
            if umin < 0:
                x[k0:k + 1] = vmin
            elif umax > 0:
                x[k0:k + 1] = vmax
            else:
                x[k0:k + 1] = vmin + umin / (k - k0 + 1)
            break
    return x


class ProxEqualityDouble(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        mean = np.mean(values)
        if self.positive and mean < 0:
            mean = 0.0
        out[idx] = mean

    def value(self, coeffs):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        if values.size == 0:
            return 0.0
        if np.allclose(values, values.flat[0]):
            return 0.0
        return sys.float_info.max


class ProxEqualityFloat(ProxEqualityDouble):
    pass


class ProxElasticNetDouble(_SeparableProx):
    def __init__(self, strength, ratio, *args, **kwargs):
        super().__init__(strength, *args, **kwargs)
        self.ratio = float(ratio)

    def set_ratio(self, ratio: float):
        self.ratio = float(ratio)

    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx].copy()
        l1_strength = self.strength * self.ratio
        l2_strength = self.strength * (1 - self.ratio)
        thresh = step * l1_strength
        values = np.sign(values) * np.maximum(np.abs(values) - thresh, 0)
        values = values / (1.0 + step * l2_strength)
        if self.positive:
            values[values < 0] = 0
        out[idx] = values

    def value(self, coeffs):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        l1_strength = self.strength * self.ratio
        l2_strength = self.strength * (1 - self.ratio)
        return float(l1_strength * np.abs(values).sum()
                     + 0.5 * l2_strength * np.linalg.norm(values) ** 2)


class ProxElasticNetFloat(ProxElasticNetDouble):
    pass


class ProxPositiveDouble(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx].copy()
        values[values < 0] = 0
        out[idx] = values

    def value(self, coeffs):
        return 0.0


class ProxPositiveFloat(ProxPositiveDouble):
    pass


class ProxZeroDouble(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        out[idx] = coeffs[idx]

    def value(self, coeffs):
        return 0.0


class ProxZeroFloat(ProxZeroDouble):
    pass


class ProxTVDouble(_SeparableProx):
    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        denoised = _tv1d_denoise(values.astype(float), step * self.strength)
        if self.positive:
            denoised[denoised < 0] = 0
        out[idx] = denoised

    def value(self, coeffs):
        idx = self._slice(coeffs)
        values = coeffs[idx]
        if values.size <= 1:
            return 0.0
        return float(self.strength * np.abs(np.diff(values)).sum())


class ProxTVFloat(ProxTVDouble):
    pass


class ProxBinarsityDouble(_SeparableProx):
    def __init__(self, strength, blocks_start, blocks_length,
                 start: Optional[int] = None, end: Optional[int] = None,
                 positive: bool = False):
        super().__init__(strength, start, end, positive)
        self.blocks_start = np.asarray(blocks_start, dtype=int)
        self.blocks_length = np.asarray(blocks_length, dtype=int)

    def set_blocks_start(self, blocks_start):
        self.blocks_start = np.asarray(blocks_start, dtype=int)

    def set_blocks_length(self, blocks_length):
        self.blocks_length = np.asarray(blocks_length, dtype=int)

    def call(self, coeffs, step, out):
        offset = self.start or 0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block_start = offset + int(start)
            block_end = block_start + int(length)
            values = coeffs[block_start:block_end].astype(float)
            denoised = _tv1d_denoise(values, step * self.strength)
            denoised -= denoised.mean()
            if self.positive:
                denoised[denoised < 0] = 0
            out[block_start:block_end] = denoised

    def value(self, coeffs):
        offset = self.start or 0
        total = 0.0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block_start = offset + int(start)
            block_end = block_start + int(length)
            values = coeffs[block_start:block_end]
            if values.size > 1:
                total += np.abs(np.diff(values)).sum()
        return float(self.strength * total)


class ProxBinarsityFloat(ProxBinarsityDouble):
    pass


def _slope_weights(strength: float, fdr: float, size: int) -> np.ndarray:
    from scipy.stats import norm as normal

    tmp = fdr / (2.0 * size)
    return strength * normal.ppf(1 - tmp * np.arange(1, size + 1))


def _prox_sorted_l1(values: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    values_sorted = values[order]
    v = values_sorted - lambdas
    n = len(v)
    if n == 0:
        return values.copy()
    x = np.zeros(n, dtype=float)
    k = 0
    while k < n:
        start = k
        total = v[k]
        while k < n - 1 and (total / (k - start + 1)) <= v[k + 1]:
            k += 1
            total += v[k]
        avg = total / (k - start + 1)
        avg = max(avg, 0.0)
        x[start:k + 1] = avg
        k += 1
    out = np.zeros(n, dtype=float)
    out[order] = x
    return out


class _SkipProx(_BaseProx):
    def __init__(self, *args, **kwargs):
        _skip()


class ProxGroupL1Double(_SeparableProx):
    def __init__(self, strength, blocks_start, blocks_length,
                 start: Optional[int] = None, end: Optional[int] = None,
                 positive: bool = False):
        super().__init__(strength, start, end, positive)
        self.blocks_start = np.asarray(blocks_start, dtype=int)
        self.blocks_length = np.asarray(blocks_length, dtype=int)

    def set_blocks_start(self, blocks_start):
        self.blocks_start = np.asarray(blocks_start, dtype=int)

    def set_blocks_length(self, blocks_length):
        self.blocks_length = np.asarray(blocks_length, dtype=int)

    def call(self, coeffs, step, out):
        offset = self.start or 0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block_start = offset + int(start)
            block_end = block_start + int(length)
            values = coeffs[block_start:block_end]
            norm = np.linalg.norm(values)
            if norm == 0:
                out[block_start:block_end] = 0
                continue
            thresh = step * self.strength * np.sqrt(values.size)
            factor = max(1 - thresh / norm, 0.0)
            out[block_start:block_end] = values * factor
            if self.positive:
                block_vals = out[block_start:block_end]
                block_vals[block_vals < 0] = 0
                out[block_start:block_end] = block_vals

    def value(self, coeffs):
        offset = self.start or 0
        total = 0.0
        for start, length in zip(self.blocks_start, self.blocks_length):
            block_start = offset + int(start)
            block_end = block_start + int(length)
            values = coeffs[block_start:block_end]
            total += self.strength * np.sqrt(values.size) * np.linalg.norm(values)
        return float(total)


class ProxGroupL1Float(ProxGroupL1Double):
    pass


class ProxMultiDouble(_BaseProx):
    def __init__(self, proxs):
        self.proxs = list(proxs)

    def call(self, coeffs, step, out):
        out[:] = coeffs
        for prox in self.proxs:
            prox.call(out, step, out)

    def value(self, coeffs):
        return float(sum(prox.value(coeffs) for prox in self.proxs))


class ProxMultiFloat(ProxMultiDouble):
    pass


class ProxSlopeDouble(_BaseProx):
    def __init__(self, strength, fdr, start: Optional[int] = None,
                 end: Optional[int] = None, positive: bool = False):
        self.strength = float(strength)
        self.fdr = float(fdr)
        self.start = start
        self.end = end
        self.positive = positive
        self._weights = None

    def set_strength(self, strength: float):
        self.strength = float(strength)
        self._weights = None

    def set_false_discovery_rate(self, fdr: float):
        self.fdr = float(fdr)
        self._weights = None

    def set_positive(self, positive: bool):
        self.positive = bool(positive)

    def set_start_end(self, start: int, end: int):
        self.start = int(start)
        self.end = int(end)
        self._weights = None

    def _slice(self, coeffs: np.ndarray):
        if self.start is None or self.end is None:
            return slice(None)
        return slice(self.start, self.end)

    def _get_weights(self, size: int) -> np.ndarray:
        if self._weights is None or self._weights.size != size:
            self._weights = _slope_weights(self.strength, self.fdr, size)
        return self._weights

    def get_weight_i(self, i: int) -> float:
        weights = self._get_weights(max(i + 1, 1))
        return float(weights[i])

    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx].copy()
        signs = np.sign(values)
        abs_vals = np.abs(values)
        lambdas = self._get_weights(abs_vals.size) * step
        prox = _prox_sorted_l1(abs_vals, lambdas)
        if self.positive:
            prox = np.maximum(prox, 0)
            signs = np.ones_like(signs)
        out[idx] = signs * prox

    def value(self, coeffs):
        idx = self._slice(coeffs)
        abs_vals = np.abs(coeffs[idx])
        weights = self._get_weights(abs_vals.size)
        sorted_vals = abs_vals[np.argsort(-abs_vals)]
        return float(np.dot(sorted_vals, weights))


class ProxSlopeFloat(ProxSlopeDouble):
    pass


class ProxSortedL1Double(_BaseProx):
    def __init__(self, strength, fdr, weights_type,
                 start: Optional[int] = None, end: Optional[int] = None,
                 positive: bool = False):
        self.strength = float(strength)
        self.fdr = float(fdr)
        self.weights_type = int(weights_type)
        self.start = start
        self.end = end
        self.positive = positive
        self._weights = None

    def set_strength(self, strength: float):
        self.strength = float(strength)
        self._weights = None

    def set_fdr(self, fdr: float):
        self.fdr = float(fdr)
        self._weights = None

    def set_weights_type(self, weights_type: int):
        self.weights_type = int(weights_type)
        self._weights = None

    def set_positive(self, positive: bool):
        self.positive = bool(positive)

    def set_start_end(self, start: int, end: int):
        self.start = int(start)
        self.end = int(end)
        self._weights = None

    def _slice(self, coeffs: np.ndarray):
        if self.start is None or self.end is None:
            return slice(None)
        return slice(self.start, self.end)

    def _get_weights(self, size: int) -> np.ndarray:
        if self._weights is None or self._weights.size != size:
            if self.weights_type == WeightsType_bh:
                self._weights = _slope_weights(self.strength, self.fdr, size)
            else:
                raise NotImplementedError("``oscar`` weights are not supported.")
        return self._weights

    def get_weight_i(self, i: int) -> float:
        weights = self._get_weights(max(i + 1, 1))
        return float(weights[i])

    def call(self, coeffs, step, out):
        idx = self._slice(coeffs)
        values = coeffs[idx].copy()
        signs = np.sign(values)
        abs_vals = np.abs(values)
        lambdas = self._get_weights(abs_vals.size) * step
        prox = _prox_sorted_l1(abs_vals, lambdas)
        if self.positive:
            prox = np.maximum(prox, 0)
            signs = np.ones_like(signs)
        out[idx] = signs * prox

    def value(self, coeffs):
        idx = self._slice(coeffs)
        abs_vals = np.abs(coeffs[idx])
        weights = self._get_weights(abs_vals.size)
        sorted_vals = abs_vals[np.argsort(-abs_vals)]
        return float(np.dot(sorted_vals, weights))


class ProxSortedL1Float(ProxSortedL1Double):
    pass
