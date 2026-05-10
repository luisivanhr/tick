"""Hawkes kernel objects."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from our_hawkes.base import BaseEstimator, TimeFunction

from .numeric import (
    exp_kernel_convolution,
    exp_kernel_primitive_convolution,
    sumexp_kernel_convolution,
    sumexp_kernel_primitive_convolution,
)


class HawkesKernel(BaseEstimator):
    """Base class for one entry of a Hawkes kernel matrix."""

    def __init__(self, support: float = math.inf):
        self.support = float(support)

    def is_zero(self) -> bool:
        return self.get_norm() == 0.0

    def get_support(self) -> float:
        return self.support

    def get_plot_support(self) -> float:
        if math.isinf(self.support):
            return 1.0
        return self.support

    def get_value(self, t: float) -> float:
        t = float(t)
        if t < 0:
            return 0.0
        if not math.isinf(self.support) and t >= self.support:
            return 0.0
        return float(self._value(t))

    def _value(self, t: float) -> float:
        del t
        return 0.0

    def get_values(self, t_values: Any) -> np.ndarray:
        arr = np.asarray(t_values, dtype=float)
        return np.vectorize(self.get_value, otypes=[float])(arr)

    def get_primitive_value(self, *args: float) -> float:
        if len(args) == 1:
            t = float(args[0])
            if t <= 0:
                return 0.0
            return float(self._primitive(t))
        if len(args) == 2:
            s, t = map(float, args)
            if t < s:
                raise ValueError("cannot compute primitive on an interval with t < s")
            return self.get_primitive_value(t - max(s, 0.0))
        raise TypeError("get_primitive_value expects t or (s, t)")

    def _primitive(self, t: float) -> float:
        upper = min(t, self.support if not math.isinf(self.support) else t)
        if upper <= 0:
            return 0.0
        xs = np.linspace(0.0, upper, 2048)
        ys = self.get_values(xs)
        return float(np.trapezoid(ys, xs))

    def get_primitive_values(self, t_values: Any) -> np.ndarray:
        arr = np.asarray(t_values, dtype=float)
        return np.vectorize(self.get_primitive_value, otypes=[float])(arr)

    def get_norm(self, n_steps: int = 10000) -> float:
        if self.support == 0:
            return 0.0
        if math.isinf(self.support):
            plot_support = self.get_plot_support()
            upper = max(float(plot_support), 1.0)
        else:
            upper = self.support
        xs = np.linspace(0.0, upper, n_steps)
        ys = np.abs(self.get_values(xs))
        return float(np.trapezoid(ys, xs))

    def get_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return float(sum(self.get_value(time - tk) for tk in timestamps if tk < time))

    def get_primitive_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return float(
            sum(self.get_primitive_value(time - tk) for tk in timestamps if tk < time)
        )

    def future_bound(self, current_lag: float | None = None) -> float:
        del current_lag
        support = self.get_plot_support()
        if math.isinf(support) or support <= 0:
            support = 1.0
        xs = np.linspace(0.0, support, 512)
        return float(max(np.max(self.get_values(xs)), 0.0))

    def __str__(self) -> str:
        return "Kernel"

    def __repr__(self) -> str:
        return str(self)

    def __strtex__(self) -> str:
        return str(self)


class HawkesKernel0(HawkesKernel):
    """Zero Hawkes kernel."""

    def __init__(self):
        super().__init__(support=0.0)

    def is_zero(self) -> bool:
        return True

    def get_norm(self, n_steps: int = 10000) -> float:
        del n_steps
        return 0.0

    def get_plot_support(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return "0"

    def __repr__(self) -> str:
        return "0"

    def __strtex__(self) -> str:
        return r"$0$"


class HawkesKernelExp(HawkesKernel):
    """Exponential kernel ``alpha * beta * exp(-beta t)``."""

    def __init__(self, intensity: float, decay: float):
        if decay < 0:
            raise ValueError("decay must be non-negative")
        super().__init__(support=math.inf)
        self.intensity = float(intensity)
        self.decay = float(decay)

    def _value(self, t: float) -> float:
        if self.intensity == 0.0 or self.decay == 0.0:
            return 0.0
        return self.intensity * self.decay * math.exp(-self.decay * t)

    def _primitive(self, t: float) -> float:
        if self.intensity == 0.0 or self.decay == 0.0:
            return 0.0
        return self.intensity * (1.0 - math.exp(-self.decay * t))

    def get_norm(self, n_steps: int = 10000) -> float:
        del n_steps
        return float(self.intensity)

    def get_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return exp_kernel_convolution(time, timestamps, self.intensity, self.decay)

    def get_primitive_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return exp_kernel_primitive_convolution(time, timestamps, self.intensity, self.decay)

    def get_plot_support(self) -> float:
        if self.decay <= 0:
            return 0.0
        return float(self.intensity / self.decay)

    def future_bound(self, current_lag: float | None = None) -> float:
        if self.intensity < 0:
            return 0.0
        lag = 0.0 if current_lag is None else max(float(current_lag), 0.0)
        return self.get_value(lag)

    def __str__(self) -> str:
        if self.intensity == 0:
            return "0"
        if self.decay == 0:
            return f"{self.intensity:g}"
        return f"{self.intensity:g} * {self.decay:g} * exp(- {self.decay:g} * t)"

    def __repr__(self) -> str:
        return str(self).replace(" ", "")

    def __strtex__(self) -> str:
        if self.intensity == 0:
            return r"$0$"
        if self.decay == 0:
            return rf"${self.intensity:g}$"
        coefficient = self.intensity * self.decay
        coefficient_prefix = "" if coefficient == 1 else f"{coefficient:g} "
        if self.decay == 1:
            if coefficient == 1:
                return r"$e^{-t}$"
            return rf"${coefficient_prefix}e^{{- t}}$"
        return rf"${coefficient_prefix}e^{{-{self.decay:g} t}}$"


class HawkesKernelSumExp(HawkesKernel):
    """Sum of exponential kernels."""

    def __init__(self, intensities: Any, decays: Any):
        intensities = np.asarray(intensities, dtype=float)
        decays = np.asarray(decays, dtype=float)
        if intensities.ndim != 1 or decays.ndim != 1:
            raise ValueError("intensities and decays must be one-dimensional")
        if intensities.size == 0:
            raise ValueError("intensities and decays must not be empty")
        if intensities.size != decays.size:
            raise ValueError("intensities and decays must have the same length")
        if np.any(decays < 0):
            raise ValueError("decays must be non-negative")
        super().__init__(support=math.inf)
        self.intensities = np.ascontiguousarray(intensities, dtype=float)
        self.decays = np.ascontiguousarray(decays, dtype=float)

    @property
    def n_decays(self) -> int:
        return int(self.decays.size)

    def _value(self, t: float) -> float:
        value = 0.0
        for intensity, decay in zip(self.intensities, self.decays):
            if intensity != 0.0 and decay != 0.0:
                value += intensity * decay * math.exp(-decay * t)
        return float(value)

    def _primitive(self, t: float) -> float:
        value = 0.0
        for intensity, decay in zip(self.intensities, self.decays):
            if intensity != 0.0 and decay != 0.0:
                value += intensity * (1.0 - math.exp(-decay * t))
        return float(value)

    def get_norm(self, n_steps: int = 10000) -> float:
        del n_steps
        return float(np.sum(self.intensities))

    def get_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return sumexp_kernel_convolution(time, timestamps, self.intensities, self.decays)

    def get_primitive_convolution(self, time: float, timestamps: np.ndarray) -> float:
        return sumexp_kernel_primitive_convolution(time, timestamps, self.intensities, self.decays)

    def get_plot_support(self) -> float:
        positive = self.decays[self.decays > 0]
        if positive.size == 0:
            return 1.0
        return float(3.0 / np.min(positive))

    def _generate_corresponding_single_exp_kernels(self) -> list[HawkesKernelExp]:
        return [
            HawkesKernelExp(float(intensity), float(decay))
            for intensity, decay in zip(self.intensities, self.decays)
        ]

    def __str__(self) -> str:
        return " + ".join(str(k) for k in self._generate_corresponding_single_exp_kernels())

    def __repr__(self) -> str:
        return " + ".join(
            repr(k) for k in self._generate_corresponding_single_exp_kernels()
        )

    def __strtex__(self) -> str:
        return " + ".join(
            k.__strtex__() for k in self._generate_corresponding_single_exp_kernels()
        )


class HawkesKernelPowerLaw(HawkesKernel):
    """Power-law Hawkes kernel ``multiplier * (cutoff + t) ** -exponent``."""

    def __init__(
        self,
        multiplier: float,
        cutoff: float,
        exponent: float,
        support: float = -1.0,
        error: float = 1e-5,
    ):
        self.multiplier = float(multiplier)
        self.cutoff = float(cutoff)
        self.exponent = float(exponent)
        if self.cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if self.exponent < 0:
            raise ValueError("exponent must be non-negative")
        if support <= 0:
            if error <= 0:
                raise ValueError("either support or error must be positive")
            if self.multiplier <= 0 or self.exponent <= 0:
                support = 0.0
            else:
                support = max((self.multiplier / error) ** (1.0 / self.exponent) - self.cutoff, 0.0)
        super().__init__(support=float(support))

    def _value(self, t: float) -> float:
        if self.multiplier == 0:
            return 0.0
        return self.multiplier * (self.cutoff + t) ** (-self.exponent)

    def _primitive(self, t: float) -> float:
        upper = min(t, self.support)
        if upper <= 0 or self.multiplier == 0:
            return 0.0
        if self.exponent == 1:
            return self.multiplier * math.log((self.cutoff + upper) / self.cutoff)
        return (
            self.multiplier
            * ((self.cutoff + upper) ** (1.0 - self.exponent) - self.cutoff ** (1.0 - self.exponent))
            / (1.0 - self.exponent)
        )

    def get_norm(self, n_steps: int = 10000) -> float:
        del n_steps
        return self._primitive(self.support)

    def __str__(self) -> str:
        if self.multiplier == 0:
            return "0"
        if self.exponent == 0:
            return f"{self.multiplier:g}"
        return f"{self.multiplier:g} * ({self.cutoff:g} + t)^(-{self.exponent:g})"

    def __repr__(self) -> str:
        return str(self).replace(" ", "")

    def __strtex__(self) -> str:
        if self.multiplier == 0:
            return r"$0$"
        if self.exponent == 0:
            return rf"${self.multiplier:g}$"
        if self.multiplier == 1:
            return rf"$({self.cutoff:g}+t)^{{-{self.exponent:g}}}$"
        return rf"${self.multiplier:g} ({self.cutoff:g}+t)^{{-{self.exponent:g}}}$"


class HawkesKernelTimeFunc(HawkesKernel):
    """Kernel defined by a :class:`~our_hawkes.base.TimeFunction`."""

    def __init__(
        self,
        time_function: TimeFunction | None = None,
        t_values: Any | None = None,
        y_values: Any | None = None,
    ):
        if (time_function is None) == (t_values is None):
            raise ValueError("either time_function or t_values/y_values must be provided")
        if t_values is not None and y_values is None:
            raise ValueError("t_values and y_values must both be provided")
        if time_function is None:
            time_function = TimeFunction(
                (t_values, y_values),
                border_type=TimeFunction.Border0,
                inter_mode=TimeFunction.InterConstRight,
            )
        self.time_function = time_function
        support = float(time_function.original_t[-1]) if not time_function.is_constant else math.inf
        super().__init__(support=support)

    def _value(self, t: float) -> float:
        return float(self.time_function.value(t))

    def _primitive(self, t: float) -> float:
        return self.time_function.primitive(min(t, self.support))

    def get_norm(self, n_steps: int = 10000) -> float:
        del n_steps
        return float(self.time_function.get_norm())

    def future_bound(self, current_lag: float | None = None) -> float:
        return self.time_function.future_bound(0.0 if current_lag is None else current_lag)

    def __str__(self) -> str:
        return "KernelTimeFunc"

    def __repr__(self) -> str:
        return "KernelTimeFunc"

    def __strtex__(self) -> str:
        return "TimeFunc Kernel"
