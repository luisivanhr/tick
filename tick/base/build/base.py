# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
from scipy import special as sp_special


def standard_normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal cumulative distribution function."""
    return sp_special.ndtr(x)


def standard_normal_inv_cdf(
    x: Union[float, np.ndarray],
    out: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Inverse standard normal CDF.

    If ``out`` is provided, it will be filled in-place and returned.
    """
    result = sp_special.ndtri(x)
    if out is not None:
        out[...] = result
        return out
    return result


def throw_out_of_range() -> None:
    raise IndexError("out_of_range")


def throw_system_error() -> None:
    import errno
    import os

    raise RuntimeError(os.strerror(errno.EACCES))


def throw_invalid_argument() -> None:
    raise ValueError("invalid_argument")


def throw_domain_error() -> None:
    raise ValueError("domain_error")


def throw_runtime_error() -> None:
    raise RuntimeError("runtime_error")


def throw_string() -> None:
    raise RuntimeError("string")


class A0:
    """Lightweight stand-in for the C++ test class used in unit tests."""

    def __init__(self) -> None:
        self._cpp_int = 0

    def set_cpp_int(self, value: int) -> None:
        self._cpp_int = value

    def get_cpp_int(self) -> int:
        return self._cpp_int


@dataclass
class _TimeFunctionState:
    t_values: Optional[np.ndarray]
    y_values: Optional[np.ndarray]
    is_constant: bool
    constant_value: float
    border_type: int
    inter_mode: int
    dt: float
    border_value: float


class TimeFunction:
    """Pure-Python implementation of the TimeFunction C++ helper."""

    InterMode_InterLinear = 0
    InterMode_InterConstLeft = 1
    InterMode_InterConstRight = 2

    BorderType_Border0 = 0
    BorderType_BorderConstant = 1
    BorderType_BorderContinue = 2
    BorderType_Cyclic = 3

    def __init__(
        self,
        *args: Union[float, np.ndarray],
    ) -> None:
        if len(args) == 1 and isinstance(args[0], (float, int)):
            self._state = _TimeFunctionState(
                t_values=None,
                y_values=None,
                is_constant=True,
                constant_value=float(args[0]),
                border_type=self.BorderType_Border0,
                inter_mode=self.InterMode_InterLinear,
                dt=0.0,
                border_value=0.0,
            )
            return

        if len(args) < 2:
            raise ValueError("TimeFunction expects (t_values, y_values, ...)")

        t_values = np.asarray(args[0], dtype=float)
        y_values = np.asarray(args[1], dtype=float)
        border_type = int(args[2]) if len(args) > 2 else self.BorderType_Border0
        inter_mode = int(args[3]) if len(args) > 3 else self.InterMode_InterLinear
        dt = float(args[4]) if len(args) > 4 else 0.0
        border_value = float(args[5]) if len(args) > 5 else 0.0

        if t_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("t_values and y_values must be 1D arrays")
        if t_values.shape[0] != y_values.shape[0]:
            raise ValueError("t_values and y_values must be the same length")

        if dt == 0 and len(t_values) > 1:
            diffs = np.diff(np.sort(t_values))
            min_diff = float(np.min(diffs)) if diffs.size else 1.0
            dt = min_diff / 5.0 if min_diff > 0 else 1.0

        self._state = _TimeFunctionState(
            t_values=t_values,
            y_values=y_values,
            is_constant=False,
            constant_value=0.0,
            border_type=border_type,
            inter_mode=inter_mode,
            dt=dt,
            border_value=border_value,
        )

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        state = self._state
        if state.is_constant:
            return np.full_like(t, state.constant_value, dtype=float)

        t_values = state.t_values
        y_values = state.y_values
        if t_values is None or y_values is None:
            return np.zeros_like(t, dtype=float)

        t_sorted_indices = np.argsort(t_values)
        t_sorted = t_values[t_sorted_indices]
        y_sorted = y_values[t_sorted_indices]

        if state.inter_mode == self.InterMode_InterLinear:
            values = np.interp(t, t_sorted, y_sorted)
        elif state.inter_mode == self.InterMode_InterConstLeft:
            indices = np.searchsorted(t_sorted, t, side="left")
            indices = np.clip(indices, 0, len(t_sorted) - 1)
            values = y_sorted[indices]
        else:
            indices = np.searchsorted(t_sorted, t, side="right") - 1
            indices = np.clip(indices, 0, len(t_sorted) - 1)
            values = y_sorted[indices]

        min_t = t_sorted[0]
        max_t = t_sorted[-1]
        outside = (t < min_t) | (t > max_t)
        if np.any(outside):
            if state.border_type == self.BorderType_Border0:
                values = values.copy()
                values[outside] = 0.0
            elif state.border_type == self.BorderType_BorderConstant:
                values = values.copy()
                values[outside] = state.border_value
            elif state.border_type == self.BorderType_BorderContinue:
                values = values.copy()
                values[t < min_t] = y_sorted[0]
                values[t > max_t] = y_sorted[-1]
            else:
                span = max_t - min_t
                if span <= 0:
                    values = values.copy()
                    values[outside] = y_sorted[-1]
                else:
                    wrapped = ((t - min_t) % span) + min_t
                    values = self._evaluate(wrapped)
        return values

    def value(self, t: Union[float, Iterable[float], np.ndarray]) -> Union[float, np.ndarray]:
        t_array = np.asarray(t, dtype=float)
        values = self._evaluate(t_array)
        if np.isscalar(t):
            return float(values)
        return values

    def get_dt(self) -> float:
        return self._state.dt

    def get_inter_mode(self) -> int:
        return self._state.inter_mode

    def get_border_type(self) -> int:
        return self._state.border_type

    def get_border_value(self) -> float:
        return self._state.border_value

    def get_sampled_y(self) -> np.ndarray:
        state = self._state
        if state.is_constant:
            return np.array([state.constant_value], dtype=float)
        if state.t_values is None or state.y_values is None:
            return np.array([], dtype=float)
        return state.y_values.copy()

    def max_error(self, t: Union[float, Iterable[float], np.ndarray]) -> float:
        return 0.0

    def get_norm(self) -> float:
        state = self._state
        if state.is_constant:
            return math.inf
        if state.t_values is None or state.y_values is None:
            return 0.0
        return float(np.trapz(state.y_values, state.t_values))
"""Python stand-ins for legacy C++ base helpers.

The original package relied on compiled extensions to back certain
attributes. During the pure-Python rewrite we provide minimal shims so
existing tests and attribute linkage logic can continue to operate.
"""

import errno
import os
import math
from typing import Iterable

import scipy.stats


class A0:
    """Minimal counterpart to the legacy C++ A0 class."""

    def __init__(self):
        self._cpp_int = 0

    def set_cpp_int(self, value):
        self._cpp_int = value

    def get_cpp_int(self):
        return self._cpp_int


# Exception helpers ------------------------------------------------------------

def throw_out_of_range():
    raise IndexError("out_of_range")


def throw_system_error():
    raise RuntimeError(os.strerror(errno.EACCES))


def throw_invalid_argument():
    raise ValueError("invalid_argument")


def throw_domain_error():
    raise ValueError("domain_error")


def throw_runtime_error():
    raise RuntimeError("runtime_error")


def throw_string():
    raise RuntimeError("string")


# Statistics helpers -----------------------------------------------------------

def standard_normal_cdf(x: float):
    return float(scipy.stats.norm.cdf(x))


def standard_normal_inv_cdf(x, output=None):
    values = scipy.stats.norm.ppf(x)
    if output is not None:
        output[...] = values
        return output
    return float(values) if not hasattr(values, "__len__") else values


__all__: Iterable[str] = [
    "A0",
    "throw_out_of_range",
    "throw_system_error",
    "throw_invalid_argument",
    "throw_domain_error",
    "throw_runtime_error",
    "throw_string",
    "standard_normal_cdf",
    "standard_normal_inv_cdf",
]
