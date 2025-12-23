# License: BSD 3 clause
"""Pure-Python replacement for the legacy C++ TimeFunction.

The implementation preserves the public API relied on by tests while
removing the compiled dependency.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from tick.base.base import Base


class TimeFunction(Base):
    """A function depending on time.

    It is causal as its value is zero for all :math:`t < 0`.
    """

    InterLinear = 0
    InterConstLeft = 1
    InterConstRight = 2

    Border0 = 0
    BorderConstant = 1
    BorderContinue = 2
    Cyclic = 3

    _attrinfos = {
        "original_y": {"writable": False},
        "original_t": {"writable": False},
        "is_constant": {"writable": False},
        "dt": {"writable": False},
        "inter_mode": {"writable": False},
        "border_type": {"writable": False},
        "border_value": {"writable": False},
        "sampled_y": {"writable": False},
    }

    def __init__(
        self,
        values,
        border_type: int = Border0,
        inter_mode: int = InterLinear,
        dt: float = 0,
        border_value: float = 0,
    ):
        Base.__init__(self)

        self.border_type = border_type
        self.inter_mode = inter_mode
        self.border_value = border_value

        if isinstance(values, (int, float)):
            self.is_constant = True
            self.constant_value = float(values)
            self.original_t = np.array([0.0, 1.0])
            self.original_y = np.array([self.constant_value, self.constant_value])
            self.dt = float(dt) if dt else 0.0
            self.sampled_y = np.array([self.constant_value])
        else:
            t_values = np.asarray(values[0], dtype=float)
            y_values = np.asarray(values[1], dtype=float)
            if t_values.ndim != 1 or y_values.ndim != 1:
                raise ValueError("TimeFunction expects 1d arrays for t and y values")
            if t_values.shape[0] != y_values.shape[0]:
                raise ValueError("t_values and y_values must have the same length")
            if np.any(np.diff(t_values) < 0):
                raise ValueError("t_values must be sorted in increasing order")

            self.is_constant = False
            self.original_t = t_values
            self.original_y = y_values
            self.dt = float(dt) if dt else float(np.min(np.diff(t_values)) / 5.0)
            grid = np.arange(t_values[0], t_values[-1] + self.dt, self.dt)
            self.sampled_y = self.value(grid)

    def _wrap_time(self, t: np.ndarray) -> np.ndarray:
        period = self.original_t[-1] - self.original_t[0]
        if period == 0:
            return np.full_like(t, self.original_t[0])
        return self.original_t[0] + np.mod(t - self.original_t[0], period)

    def _interpolate(self, t: np.ndarray) -> np.ndarray:
        t_values = self.original_t
        y_values = self.original_y

        if self.border_type == self.Cyclic:
            t = self._wrap_time(t)
            border_type = self.Border0
        else:
            border_type = self.border_type

        before_first = t < t_values[0]
        after_last = t > t_values[-1]
        inside = ~(before_first | after_last)

        result = np.zeros_like(t, dtype=float)

        if border_type == self.BorderConstant:
            result[before_first] = self.border_value
            result[after_last] = self.border_value
        elif border_type == self.BorderContinue:
            result[before_first] = y_values[0]
            result[after_last] = y_values[-1]

        if not np.any(inside):
            return result

        if self.inter_mode == self.InterLinear:
            result[inside] = np.interp(t[inside], t_values, y_values)
        elif self.inter_mode == self.InterConstLeft:
            idx_right = np.searchsorted(t_values, t[inside], side="left")
            idx_right = np.clip(idx_right, 0, len(y_values) - 1)
            result[inside] = y_values[idx_right]
        elif self.inter_mode == self.InterConstRight:
            idx_left = np.searchsorted(t_values, t[inside], side="right") - 1
            idx_left = np.clip(idx_left, 0, len(y_values) - 1)
            result[inside] = y_values[idx_left]
        else:
            raise ValueError(f"Unknown interpolation mode: {self.inter_mode}")

        return result

    def value(self, t):
        """Gives the value of the TimeFunction at provided time."""
        if self.is_constant:
            return np.asarray(t, dtype=float) * 0 + self.constant_value if isinstance(t, np.ndarray) else self.constant_value

        t_array = np.asarray(t, dtype=float)
        values = self._interpolate(t_array)
        return values if isinstance(t, np.ndarray) else float(values)

    def _max_error(self, t):
        return abs(self.dt)

    def get_norm(self):
        """Computes the integral value of the TimeFunction."""
        if self.is_constant:
            return self.constant_value * (self.original_t[-1] - self.original_t[0])

        t_start, t_end = self.original_t[0], self.original_t[-1]
        n_steps = int(np.ceil((t_end - t_start) / self.dt))

        if self.inter_mode == self.InterConstLeft:
            evaluation_points = t_start + (np.arange(n_steps) + 0.5) * self.dt
            return float(np.sum(self.value(evaluation_points)) * self.dt)
        elif self.inter_mode == self.InterConstRight:
            evaluation_points = t_start + np.arange(n_steps) * self.dt
            return float(np.sum(self.value(evaluation_points)) * self.dt)
        else:
            evaluation_points = t_start + (np.arange(n_steps) + 0.5) * self.dt
            raw = np.sum(self.value(evaluation_points)) * self.dt
            adjust = raw * (t_end - t_start) / (t_end - t_start + self.dt)
            return float(adjust)
