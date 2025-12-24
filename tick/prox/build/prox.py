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
"""Pure-Python proximal operator implementations used during the rewrite.

These classes mimic the C++ bindings that previously lived in :mod:`tick.prox`
so that the high level Python APIs remain importable while the algorithms are
re-implemented.
"""
from __future__ import annotations

import sys
import numpy as np


class _RangeMixin:
    """Utility mixin that stores and updates the active range."""

    def __init__(self, start: int | None = None, end: int | None = None):
        self.start = start
        self.end = end

    def set_start_end(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    def _slice(self, array: np.ndarray) -> np.ndarray:
        if self.start is None:
            return array
        return array[self.start : self.end]

    def _apply_slice(self, source: np.ndarray, dest: np.ndarray) -> None:
        if self.start is None:
            dest[:] = source
        else:
            dest[self.start : self.end] = source


class ProxZeroDouble(_RangeMixin):
    def __init__(self, strength: float, start: int | None = None, end: int | None = None):
        super().__init__(start, end)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs

    def value(self, coeffs: np.ndarray) -> float:
        return 0.0


ProxZeroFloat = ProxZeroDouble


class ProxPositiveDouble(_RangeMixin):
    def __init__(self, strength: float, start: int | None = None, end: int | None = None):
        super().__init__(start, end)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = out if self.start is None else out[self.start : self.end]
        np.maximum(target, 0, out=target)

    def value(self, coeffs: np.ndarray) -> float:
        return 0.0


ProxPositiveFloat = ProxPositiveDouble


class ProxL1Double(_RangeMixin):
    def __init__(self, strength: float, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        step_arr = step if np.ndim(step) else None
        thresh = step * self.strength if step_arr is None else np.asarray(step) * self.strength
        shrunk = np.sign(target) * np.maximum(np.abs(target) - thresh, 0)
        if self.positive:
            np.maximum(shrunk, 0, out=shrunk)
        target[:] = shrunk
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        vals = np.abs(self._slice(coeffs))
        return float(self.strength * vals.sum())


ProxL1Float = ProxL1Double


class ProxL1wDouble(_RangeMixin):
    def __init__(self, strength: float, weights: np.ndarray, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 2:
            start, end = args
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.weights = np.asarray(weights)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        weights = self.weights if self.start is None else self.weights
        thresh = step * self.strength * weights
        shrunk = np.sign(target) * np.maximum(np.abs(target) - thresh, 0)
        if self.positive:
            np.maximum(shrunk, 0, out=shrunk)
        target[:] = shrunk
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        vals = np.abs(self._slice(coeffs))
        weights = self.weights if self.start is None else self.weights
        return float(self.strength * (weights * vals).sum())


ProxL1wFloat = ProxL1wDouble


class ProxL2Double(_RangeMixin):
    def __init__(self, strength: float, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        n = target.size
        norm = np.linalg.norm(target)
        if norm == 0:
            shrink = 0.0
        else:
            shrink = max(0.0, 1.0 - step * self.strength * np.sqrt(n) / norm)
        target *= shrink
        if self.positive:
            np.maximum(target, 0, out=target)
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        target = self._slice(coeffs)
        return float(self.strength * np.sqrt(target.size) * np.linalg.norm(target))


ProxL2Float = ProxL2Double


class ProxL2SqDouble(_RangeMixin):
    def __init__(self, strength: float, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        scale = 1.0 / (1.0 + step * self.strength)
        target *= scale
        if self.positive:
            np.maximum(target, 0, out=target)
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        target = self._slice(coeffs)
        return float(0.5 * self.strength * np.linalg.norm(target) ** 2)


ProxL2sqDouble = ProxL2SqDouble
ProxL2SqFloat = ProxL2SqDouble
ProxL2sqFloat = ProxL2SqDouble


class ProxElasticNetDouble(_RangeMixin):
    def __init__(self, strength: float, ratio: float, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.ratio = float(ratio)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        # Apply L1 then L2sq
        l1 = ProxL1Double(self.ratio * self.strength, self.start, self.end, self.positive)
        l1.start, l1.end = self.start, self.end
        l1.set_positive(self.positive)
        l1.call(target, step, target)
        l2 = ProxL2SqDouble((1 - self.ratio) * self.strength, self.start, self.end, self.positive)
        l2.start, l2.end = self.start, self.end
        l2.call(target, step, target)
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        target = self._slice(coeffs)
        l1_val = self.ratio * self.strength * np.abs(target).sum()
        l2_val = 0.5 * (1 - self.ratio) * self.strength * np.linalg.norm(target) ** 2
        return float(l1_val + l2_val)


ProxElasticNetFloat = ProxElasticNetDouble


class ProxEqualityDouble(_RangeMixin):
    def __init__(self, strength: float, start: int | None = None, end: int | None = None, positive: bool = False):
        super().__init__(start, end)
        self.strength = None
        self.positive = bool(positive)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: object = 1.0, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty_like(coeffs)
        out[:] = coeffs
        target = self._slice(out)
        mean_val = target.mean()
        if self.positive and mean_val < 0:
            mean_val = 0.0
        target[:] = mean_val
        self._apply_slice(target, out)
        return out

    def value(self, coeffs: np.ndarray) -> float:
        target = self._slice(coeffs)
        if target.size == 0:
            return 0.0
        mean_val = target.mean()
        if self.positive and mean_val < 0:
            return sys.float_info.max
        if np.allclose(target, mean_val):
            return 0.0
        return sys.float_info.max


ProxEqualityFloat = ProxEqualityDouble


class ProxGroupL1Double(_RangeMixin):
    def __init__(
        self,
        strength: float,
        blocks_start,
        blocks_length,
        start: int | None = None,
        end: int | None = None,
        positive: bool = False,
    ):
        super().__init__(start, end)
        self.strength = float(strength)
        self.blocks_start = np.asarray(blocks_start, dtype=int)
        self.blocks_length = np.asarray(blocks_length, dtype=int)
        self.positive = bool(positive)

    def set_blocks_start(self, blocks_start):
        blocks_start = np.asarray(blocks_start, dtype=int)
        if blocks_start.shape != self.blocks_length.shape:
            raise ValueError("blocks_length and blocks_start must have the same size")
        self.blocks_start = blocks_start

    def set_blocks_length(self, blocks_length):
        blocks_length = np.asarray(blocks_length, dtype=int)
        if blocks_length.shape != self.blocks_start.shape:
            raise ValueError("blocks_length and blocks_start must have the same size")
        self.blocks_length = blocks_length

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        out[:] = coeffs
        for start, length in zip(self.blocks_start, self.blocks_length):
            g_start = start if self.start is None else self.start + start
            g_end = g_start + length
            target = out[g_start:g_end]
            norm = np.linalg.norm(target)
            if norm == 0:
                shrink = 0.0
            else:
                shrink = max(0.0, 1.0 - step * self.strength * np.sqrt(length) / norm)
            target *= shrink
            if self.positive:
                np.maximum(target, 0, out=target)

    def value(self, coeffs: np.ndarray) -> float:
        total = 0.0
        for start, length in zip(self.blocks_start, self.blocks_length):
            g_start = start if self.start is None else self.start + start
            g_end = g_start + length
            sub = coeffs[g_start:g_end]
            total += self.strength * np.sqrt(length) * np.linalg.norm(sub)
        return float(total)


ProxGroupL1Float = ProxGroupL1Double


class ProxTVDouble(_RangeMixin):
    def __init__(self, strength: float, *args):
        positive = False
        start = end = None
        if len(args) == 1:
            positive = bool(args[0])
        elif len(args) == 3:
            start, end, positive = args
        super().__init__(start, end)
        self.strength = float(strength)
        self.positive = bool(positive)

    def set_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def set_positive(self, positive: bool) -> None:
        self.positive = bool(positive)

    def call(self, coeffs: np.ndarray, step: float, out: np.ndarray) -> None:
        out[:] = coeffs
        target = self._slice(out)
        result = _prox_tv1d(target, step * self.strength)
        if self.positive:
            np.maximum(result, 0, out=result)
        target[:] = result
        self._apply_slice(target, out)

    def value(self, coeffs: np.ndarray) -> float:
        target = self._slice(coeffs)
        if target.size <= 1:
            return 0.0
        return float(self.strength * np.abs(np.diff(target)).sum())


ProxTVFloat = ProxTVDouble


def _prox_tv1d(signal: np.ndarray, lam: float) -> np.ndarray:
    # Condat 2013 algorithm for 1D total-variation denoising
    n = signal.size
    if n == 0:
        return signal.copy()
    x = np.empty_like(signal)
    k = k0 = 0
    vmin = signal[0] - lam
    vmax = signal[0] + lam
    umin = lam
    umax = -lam
    for i in range(1, n):
        val = signal[i]
        umin += val - vmin
        umax += val - vmax
        if umin < -lam:
            while k0 <= k:
                x[k0] = vmin
                k0 += 1
            vmax = vmin = val - lam
            umin = lam
            umax = -lam
            k = k0 = i
            continue
        if umax > lam:
            while k0 <= k:
                x[k0] = vmax
                k0 += 1
            vmax = vmin = val + lam
            umin = lam
            umax = -lam
            k = k0 = i
            continue
        if umin >= lam:
            vmin += (umin - lam) / (i - k0 + 1)
            umin = lam
        if umax <= -lam:
            vmax += (umax + lam) / (i - k0 + 1)
            umax = -lam
        k = i
    vbar = vmin + umin / (k - k0 + 1)
    vbar = (vbar + vmax - umax / (k - k0 + 1)) / 2
    while k0 <= k:
        x[k0] = vbar
        k0 += 1
    return x


# Placeholders for unported operators
class _NotImplementedProx(_RangeMixin):
    def __init__(self, strength=0.0, fdr=None, *args, **kwargs):
        super().__init__()
        self.strength = strength
        self.fdr = fdr
        self.positive = kwargs.get("positive", False)

    def set_strength(self, val):
        self.strength = val

    def set_false_discovery_rate(self, val):
        self.fdr = val

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        np.copyto(out, coeffs)

    def value(self, coeffs: np.ndarray) -> float:
        return 0.0
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, coeffs: np.ndarray, step: object, out: np.ndarray) -> None:
        raise NotImplementedError("Proximal operator not yet ported to Python")

    def value(self, coeffs: np.ndarray) -> float:
        raise NotImplementedError("Proximal operator not yet ported to Python")


ProxMultiDouble = ProxMultiFloat = _NotImplementedProx
ProxSlopeDouble = ProxSlopeFloat = _NotImplementedProx
ProxBinarsityDouble = ProxBinarsityFloat = _NotImplementedProx
ProxSortedL1Double = ProxSortedL1Float = _NotImplementedProx
WeightsType_bh = 0
WeightsType_oscar = 1
ProxNuclearDouble = ProxNuclearFloat = _NotImplementedProx
