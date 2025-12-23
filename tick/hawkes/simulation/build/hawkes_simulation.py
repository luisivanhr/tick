"""Placeholder Python implementations for Hawkes simulation backends.

These classes keep the public API available while the C++ extensions are
rewritten in pure Python. The implementations are intentionally lightweight
and are not intended to match the performance or statistical fidelity of the
original backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from tick.base import Base, TimeFunction


class _BasePointProcess(Base):
    """Minimal point process backend used during the rewrite."""

    _attrinfos = {
        "_n_nodes": {"writable": False},
        "_seed": {"writable": True},
        "_time": {"writable": True},
        "_max_jumps": {"writable": True},
        "_timestamps": {"writable": True},
        "_itr_on": {"writable": True},
        "_itr_step": {"writable": True},
        "_itr_times": {"writable": True},
        "_itr_values": {"writable": True},
        "_compensator": {"writable": True},
        "_threshold_negative_intensity": {"writable": True},
    }

    def __init__(self, n_nodes: int, seed: int | None = None):
        super().__init__()
        self._n_nodes = int(n_nodes)
        self._seed = -1 if seed is None else int(seed)
        self._time = 0.0
        self._max_jumps = None
        self._timestamps: List[np.ndarray] = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        self._itr_on = False
        self._itr_step = -1.0
        self._itr_times = np.array([], dtype=float)
        self._itr_values: List[np.ndarray] = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        self._compensator: List[np.ndarray] = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        self._threshold_negative_intensity = False

    # Common accessors -----------------------------------------------------------------
    def get_n_nodes(self) -> int:
        return self._n_nodes

    def get_time(self) -> float:
        return float(self._time)

    def get_n_total_jumps(self) -> int:
        return int(sum(len(ts) for ts in self._timestamps))

    def get_timestamps(self) -> List[np.ndarray]:
        return self._timestamps

    def get_itr(self) -> List[np.ndarray]:
        return self._itr_values

    def get_itr_times(self) -> np.ndarray:
        return self._itr_times

    def get_itr_step(self) -> float:
        return float(self._itr_step)

    def get_ctr(self) -> List[np.ndarray]:
        return self._compensator

    def set_threshold_negative_intensity(self, allow: bool):
        self._threshold_negative_intensity = bool(allow)

    def get_seed(self) -> int:
        return self._seed

    def reseed_random_generator(self, seed: int):
        self._seed = int(seed)

    # Simulation hooks -----------------------------------------------------------------
    def activate_itr(self, step: float = -1.0):
        self._itr_on = step is not None and step > 0
        self._itr_step = float(step)

    def itr_on(self) -> bool:
        return bool(self._itr_on)

    def set_max_jumps(self, n_jumps: int):
        self._max_jumps = None if n_jumps is None else int(n_jumps)

    def simulate(self, end_time: float | int, max_jumps: int | None = None):
        """Populate timestamps with a deterministic empty simulation.

        The placeholder keeps state consistent without generating events.
        """
        if isinstance(end_time, (float, np.floating)):
            self._time = float(end_time)
        else:
            # When called with an integer, treat it as max_jumps-style value
            self._time = float(self._time)
        if max_jumps is not None:
            self._max_jumps = int(max_jumps)
        # Leave timestamps empty to avoid introducing synthetic behaviour
        self._timestamps = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        if self._itr_on and self._itr_step > 0 and self._time > 0:
            steps = int(self._time // self._itr_step) + 1
            self._itr_times = np.linspace(0.0, self._time, steps)
            self._itr_values = [np.zeros_like(self._itr_times) for _ in range(self._n_nodes)]
        return self

    def reset(self):
        self._time = 0.0
        self._timestamps = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        self._compensator = [np.array([], dtype=float) for _ in range(self._n_nodes)]
        if self._itr_on:
            self._itr_times = np.array([], dtype=float)
            self._itr_values = [np.array([], dtype=float) for _ in range(self._n_nodes)]

    # Hooks for subclasses ------------------------------------------------------
    def _intensity_at_times(self, times: np.ndarray) -> np.ndarray:
        """Return intensities for each node at the provided times.

        Subclasses override this to provide model-specific behaviour. The
        default returns zeros which keeps placeholder simulators compatible
        before their logic is implemented.
        """

        times = np.asarray(times, dtype=float)
        return np.zeros((self._n_nodes, times.shape[0]), dtype=float)

    def _update_compensator(self):
        # Basic cumulative counts used to satisfy downstream accessors
        self._compensator = [np.arange(1, len(ts) + 1, dtype=float) for ts in self._timestamps]

    def _refresh_tracked_intensity(self):
        if not (self._itr_on and self._itr_step > 0 and self._time > 0):
            return
        steps = int(self._time // self._itr_step) + 1
        self._itr_times = np.linspace(0.0, self._time, steps)
        intensities = self._intensity_at_times(self._itr_times)
        self._itr_values = [intensities[i] for i in range(self._n_nodes)]

    def store_compensator_values(self):
        self._update_compensator()

    def set_timestamps(self, timestamps: Sequence[np.ndarray], end_time: float):
        self._timestamps = [np.asarray(ts, dtype=float) for ts in timestamps]
        self._time = float(end_time)
        self._update_compensator()
        self._refresh_tracked_intensity()


class Poisson(_BasePointProcess):
    _attrinfos = {"_intensities": {"writable": False}}
    """Placeholder homogeneous Poisson simulator."""

    def __init__(self, intensities: float | Sequence[float], seed: int | None = None):
        self._intensities = np.asarray(intensities, dtype=float)
        if self._intensities.ndim == 0:
            self._intensities = self._intensities.reshape(1)
        super().__init__(self._intensities.shape[0], seed=seed)

    def get_intensities(self) -> np.ndarray:
        return self._intensities

    def _intensity_at_times(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return np.tile(self._intensities[:, None], (1, times.shape[0]))

    def simulate(self, end_time: float | int | None = None, max_jumps: int | None = None):
        rng = np.random.default_rng(None if self._seed == -1 else self._seed)
        if max_jumps is None and end_time is not None and not isinstance(end_time, (float, np.floating)):
            max_jumps = int(end_time)
            end_time = None
        end_time = float(end_time) if end_time is not None else None

        if end_time is not None:
            timestamps = []
            for lam in self._intensities:
                n_events = rng.poisson(lam * end_time) if end_time > 0 else 0
                ts = np.sort(rng.uniform(0.0, end_time, size=n_events)) if n_events > 0 else np.array([], dtype=float)
                timestamps.append(ts)
            self._timestamps = timestamps
            if max_jumps is not None:
                # Trim to the requested cap while preserving chronological order
                total = sum(len(ts) for ts in timestamps)
                if total > max_jumps:
                    flat = np.concatenate([np.column_stack((np.full(len(ts), i), ts)) for i, ts in enumerate(timestamps)])
                    order = np.argsort(flat[:, 1])[:max_jumps]
                    flat = flat[order]
                    trimmed = [flat[flat[:, 0] == i][:, 1] for i in range(self._n_nodes)]
                    self._timestamps = [np.asarray(t, dtype=float) for t in trimmed]
            self._time = end_time
        elif max_jumps is not None:
            total_rate = float(self._intensities.sum())
            if total_rate <= 0:
                return self
            timestamps = [list() for _ in range(self._n_nodes)]
            time = 0.0
            for _ in range(int(max_jumps)):
                time += rng.exponential(1.0 / total_rate)
                node = rng.choice(self._n_nodes, p=self._intensities / total_rate)
                timestamps[node].append(time)
            self._timestamps = [np.array(ts, dtype=float) for ts in timestamps]
            self._time = time
        else:
            return self

        self._update_compensator()
        self._refresh_tracked_intensity()
        return self


class InhomogeneousPoisson(_BasePointProcess):
    _attrinfos = {"_time_functions": {"writable": False}}
    """Placeholder inhomogeneous Poisson simulator."""

    def __init__(self, time_functions: Sequence[TimeFunction], seed: int | None = None):
        self._time_functions = list(time_functions)
        super().__init__(len(self._time_functions), seed=seed)

    def intensity_value(self, node: int, times):
        tf = self._time_functions[node]
        return tf.evaluate(times)

    def _intensity_at_times(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        values = np.zeros((self._n_nodes, times.shape[0]))
        for i, tf in enumerate(self._time_functions):
            values[i] = tf.evaluate(times)
        return values

    def simulate(self, end_time: float | int | None = None, max_jumps: int | None = None):
        rng = np.random.default_rng(None if self._seed == -1 else self._seed)
        if end_time is None and max_jumps is None:
            return self
        if end_time is None:
            end_time = np.inf
        else:
            end_time = float(end_time)

        # Conservative bound from provided time functions
        max_intensity = 0.0
        for tf in self._time_functions:
            if hasattr(tf, "y_values"):
                max_intensity = max(max_intensity, float(np.max(tf.y_values)))
            else:
                sample_times = np.linspace(0.0, float(end_time), num=100)
                max_intensity = max(max_intensity, float(np.max(tf.evaluate(sample_times))))
        max_intensity = max(max_intensity, 1e-12)

        timestamps: List[List[float]] = [[] for _ in range(self._n_nodes)]
        time = 0.0
        n_events = 0
        while time < end_time and (max_jumps is None or n_events < max_jumps):
            time += rng.exponential(1.0 / max_intensity)
            if time > end_time:
                break
            inst = self._intensity_at_times(np.array([time]))[:, 0]
            lambda_sum = inst.sum()
            if lambda_sum <= 0:
                continue
            if rng.uniform() * max_intensity <= lambda_sum:
                node = rng.choice(self._n_nodes, p=inst / lambda_sum)
                timestamps[node].append(time)
                n_events += 1

        self._timestamps = [np.array(ts, dtype=float) for ts in timestamps]
        self._time = float(end_time if np.isfinite(end_time) else time)
        self._update_compensator()
        self._refresh_tracked_intensity()
        return self


class Hawkes(_BasePointProcess):
    _attrinfos = {
        "_kernels": {"writable": True},
        "_baseline": {"writable": True},
    }
    """Placeholder Hawkes simulator with minimal kernel/baseline handling."""

    def __init__(self, n_nodes: int, seed: int | None = None):
        super().__init__(n_nodes, seed=seed)
        self._kernels = np.empty((n_nodes, n_nodes), dtype=object)
        self._baseline = np.zeros(n_nodes, dtype=object)

    def set_kernel(self, i: int, j: int, kernel):
        self._kernels[i, j] = kernel

    def set_baseline(self, i: int, value, values=None):
        if values is not None:
            # Piecewise constant specification
            self._baseline[i] = (np.asarray(value, dtype=float), np.asarray(values, dtype=float))
        else:
            self._baseline[i] = value

    def get_baseline(self, i: int, times=None):
        base = self._baseline[i]
        if isinstance(base, tuple):
            t_values, y_values = base
            if times is None:
                return y_values
            return np.interp(times, t_values, y_values)
        if isinstance(base, TimeFunction):
            return base.evaluate(times if times is not None else np.array([], dtype=float))
        if times is None:
            return base
        return np.full_like(times, float(base), dtype=float)

    def _intensity_at_times(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        intensities = np.zeros((self._n_nodes, times.shape[0]))
        for i in range(self._n_nodes):
            base = self.get_baseline(i, times)
            base_arr = np.asarray(base, dtype=float) if np.ndim(base) else np.full(times.shape, float(base))
            intensities[i] += base_arr
            for j in range(self._n_nodes):
                kernel = self._kernels[i, j]
                ts_j = self._timestamps[j]
                if kernel is None or ts_j is None or len(ts_j) == 0:
                    continue
                diffs = times[:, None] - ts_j[None, :]
                mask = diffs > 0
                if not np.any(mask):
                    continue
                vals = kernel.get_values(diffs)
                vals[~mask] = 0.0
                intensities[i] += vals.sum(axis=1)
            if self._threshold_negative_intensity:
                intensities[i] = np.maximum(intensities[i], 0.0)
        if not self._threshold_negative_intensity and np.any(intensities < 0):
            raise RuntimeError("Simulation stopped because intensity went negative (you could call ``threshold_negative_intensity`` to allow it)")
        return intensities

    def simulate(self, end_time: float | int | None = None, max_jumps: int | None = None):
        rng = np.random.default_rng(None if self._seed == -1 else self._seed)
        if end_time is None and max_jumps is None:
            return self
        time = float(self._time)
        end_time = float("inf") if end_time is None else float(end_time)
        timestamps: List[List[float]] = [list(ts) for ts in self._timestamps]
        total_events = sum(len(ts) for ts in timestamps)

        while time < end_time and (max_jumps is None or total_events < max_jumps):
            current_intensity = self._intensity_at_times(np.array([time]))[:, 0]
            lambda_bar = max(current_intensity.sum(), 1e-12)
            wait = rng.exponential(1.0 / lambda_bar)
            time += wait
            if time > end_time:
                break
            new_intensity = self._intensity_at_times(np.array([time]))[:, 0]
            total_intensity = new_intensity.sum()
            if total_intensity <= 0:
                continue
            accept_ratio = min(1.0, total_intensity / lambda_bar)
            if rng.uniform() <= accept_ratio:
                node = rng.choice(self._n_nodes, p=new_intensity / total_intensity)
                timestamps[node].append(time)
                total_events += 1

        self._timestamps = [np.array(ts, dtype=float) for ts in timestamps]
        self._time = time if np.isfinite(end_time) else max(time, self._time)
        self._update_compensator()
        self._refresh_tracked_intensity()
        return self


# Kernel implementations ----------------------------------------------------------------


class HawkesKernel(Base):
    _attrinfos = {"_support": {"writable": True}}

    def __init__(self):
        super().__init__()
        self._support = 0.0

    def is_zero(self) -> bool:
        return False

    def get_support(self) -> float:
        return float(self._support)

    def get_plot_support(self) -> float:
        return float(self._support if self._support else 0.0)

    def get_value(self, t: float) -> float:
        return float(self.get_values(np.array([t]))[0])

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        return np.zeros_like(t_values)

    def get_primitive_value(self, t: float) -> float:
        return float(self.get_primitive_values(np.array([t]))[0])

    def get_primitive_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        return np.zeros_like(t_values)

    def get_norm(self, n_steps: int = 10000) -> float:
        # Default Riemann approximation using primitive
        sample_times = np.linspace(0.0, self.get_plot_support() or 1.0, n_steps)
        primitive = self.get_values(sample_times)
        return float(np.trapezoid(primitive, sample_times))


class HawkesKernel0(HawkesKernel):
    def __init__(self):
        super().__init__()
        self._support = 0.0

    def is_zero(self) -> bool:
        return True


class HawkesKernelExp(HawkesKernel):
    _attrinfos = {"_support": {"writable": True}, "_intensity": {"writable": True}, "_decay": {"writable": True}}
    def __init__(self, intensity: float, decay: float):
        super().__init__()
        self._intensity = float(intensity)
        self._decay = float(decay)
        self._support = np.inf if decay == 0 else self._intensity / max(decay, 1e-12)

    def get_intensity(self) -> float:
        return self._intensity

    def get_decay(self) -> float:
        return self._decay

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        values = self._intensity * self._decay * np.exp(-self._decay * t_values)
        values[t_values < 0] = 0.0
        return values

    def get_primitive_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        prim = self._intensity * (1 - np.exp(-self._decay * np.maximum(t_values, 0)))
        return prim

    def get_norm(self, n_steps: int = 10000) -> float:
        return max(self._intensity, 0.0)


class HawkesKernelPowerLaw(HawkesKernel):
    _attrinfos = {
        "_support": {"writable": True},
        "_intensity": {"writable": True},
        "_cutoff": {"writable": True},
        "_exponent": {"writable": True},
    }
    def __init__(self, intensity: float, cutoff: float, exponent: float, support: float | None = None, *_, **__):
        super().__init__()
        self._intensity = float(intensity)
        self._cutoff = float(cutoff)
        self._exponent = float(exponent)
        self._support = np.inf if support is None else float(support)

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        values = self._intensity * (self._cutoff + np.maximum(t_values, 0)) ** (-self._exponent)
        values[t_values < 0] = 0.0
        return values

    def get_norm(self, n_steps: int = 10000) -> float:
        # Integral of cutoff^{1-exponent} / (exponent-1)
        if self._exponent <= 1:
            return np.inf
        return float(self._intensity * (self._cutoff ** (1 - self._exponent)) / (self._exponent - 1))

    def get_multiplier(self) -> float:
        return float(self._intensity)

    def get_cutoff(self) -> float:
        return float(self._cutoff)

    def get_exponent(self) -> float:
        return float(self._exponent)

    @property
    def multiplier(self) -> float:
        return self._intensity

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def exponent(self) -> float:
        return self._exponent

    def __str__(self):
        if self._intensity == 0:
            return "0"
        if self._exponent == 0:
            return str(self._intensity)
        return f"{self._intensity} * ({self._cutoff} + t)^(-{self._exponent})"

    def __repr__(self):
        if self._intensity == 0:
            return "0"
        if self._exponent == 0:
            return str(self._intensity)
        return f"{self._intensity}*({self._cutoff}+t)^(-{self._exponent})"

    def __strtex__(self):
        return self.__str__()


class HawkesKernelSumExp(HawkesKernel):
    _attrinfos = {
        "_support": {"writable": True},
        "_decays": {"writable": True},
        "_intensities": {"writable": True},
    }
    def __init__(self, intensities: Sequence[float], decays: Sequence[float], *_, **__):
        super().__init__()
        self._decays = np.asarray(decays, dtype=float)
        self._intensities = np.asarray(intensities, dtype=float)
        self._support = np.inf if len(self._decays) == 0 else 10.0 / max(self._decays.max(), 1e-12)

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        values = np.zeros_like(t_values)
        for decay, intensity in zip(self._decays, self._intensities):
            contrib = intensity * decay * np.exp(-decay * t_values)
            contrib[t_values < 0] = 0.0
            values += contrib
        return values

    def get_primitive_values(self, t_values: np.ndarray) -> np.ndarray:
        t_values = np.asarray(t_values, dtype=float)
        prim = np.zeros_like(t_values)
        for decay, intensity in zip(self._decays, self._intensities):
            prim += intensity * (1 - np.exp(-decay * np.maximum(t_values, 0)))
        return prim

    def get_norm(self, n_steps: int = 10000) -> float:
        return float(np.sum(self._intensities))

    def get_decays(self) -> np.ndarray:
        return self._decays

    def get_intensities(self) -> np.ndarray:
        return self._intensities

    def get_n_decays(self) -> int:
        return int(len(self._decays))

    @property
    def decays(self) -> np.ndarray:
        return self._decays

    @property
    def intensities(self) -> np.ndarray:
        return self._intensities

    @property
    def n_decays(self) -> int:
        return self.get_n_decays()

    def __str__(self):
        terms = []
        for intensity, decay in zip(self._intensities, self._decays):
            if decay == 0:
                terms.append(f"{intensity}")
            else:
                terms.append(f"{intensity} * {decay} * exp(- {decay} * t)")
        return " + ".join(terms)

    def __repr__(self):
        compact_terms = []
        for intensity, decay in zip(self._intensities, self._decays):
            if decay == 0:
                compact_terms.append(f"{intensity}")
            else:
                compact_terms.append(f"{intensity}*{decay}*exp(-{decay}*t)")
        return " + ".join(compact_terms)

    def __strtex__(self):
        terms = []
        for intensity, decay in zip(self._intensities, self._decays):
            coef = intensity if decay == 0 else intensity * decay
            terms.append(f"${coef} e^{{-{decay} t}}$")
        return " + ".join(terms)


class HawkesKernelTimeFunc(HawkesKernel):
    _attrinfos = {"_support": {"writable": True}, "time_function": {"writable": True}}

    def __init__(self, time_function: TimeFunction | None = None, t_values=None, y_values=None):
        super().__init__()
        if isinstance(time_function, (list, np.ndarray)) and y_values is None and t_values is not None:
            y_values = t_values
            t_values = time_function
            time_function = None
        if time_function is None:
            if t_values is None or y_values is None:
                raise ValueError("TimeFunction or (t_values, y_values) must be provided")
            time_function = TimeFunction((t_values, y_values))
        self.time_function = time_function
        self._support = time_function.t_values[-1] if len(time_function.t_values) else 0.0

    def get_values(self, t_values: np.ndarray) -> np.ndarray:
        return self.time_function.evaluate(t_values)

    def get_primitive_values(self, t_values: np.ndarray) -> np.ndarray:
        return self.time_function.primitive(t_values)

    def get_norm(self, n_steps: int = 10000) -> float:
        return float(self.time_function.primitive(self._support))

    def get_time_function(self) -> TimeFunction:
        return self.time_function

    def __str__(self):
        return "KernelTimeFunc"

    def __repr__(self):
        return "KernelTimeFunc"

    def __strtex__(self):
        return "TimeFunc Kernel"


__all__ = [
    "Hawkes",
    "Poisson",
    "InhomogeneousPoisson",
    "HawkesKernel",
    "HawkesKernel0",
    "HawkesKernelExp",
    "HawkesKernelPowerLaw",
    "HawkesKernelSumExp",
    "HawkesKernelTimeFunc",
]
