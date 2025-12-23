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
