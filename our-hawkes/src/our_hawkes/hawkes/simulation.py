"""Point-process and Hawkes simulation classes."""

from __future__ import annotations

import copy
import math
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from typing import Any

import numpy as np

from our_hawkes.base import BaseEstimator, TimeFunction, now_string

from .kernels import (
    HawkesKernel,
    HawkesKernel0,
    HawkesKernelExp,
    HawkesKernelSumExp,
)
from .numeric import exp_kernel_convolution, sumexp_kernel_convolution


class Simu(BaseEstimator):
    """Base simulation class."""

    def __init__(self, seed: int | None = None, verbose: bool = True):
        self._seed: int | None = None
        self._rng = np.random.default_rng()
        self.seed = seed
        self.verbose = verbose
        self.time_start: str | None = None
        self.time_end: str | None = None
        self.time_elapsed: float | None = None

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        self._seed = None if value is None else int(value)
        rng_seed = None if self._seed is None or self._seed < 0 else self._seed
        self._rng = np.random.default_rng(rng_seed)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _start_simulation(self) -> None:
        self.time_start = now_string()
        self._time_start = time.time()
        if self.verbose:
            print(f"Launching simulation using {self.name}...")

    def _end_simulation(self) -> None:
        self.time_end = now_string()
        self.time_elapsed = time.time() - self._time_start
        if self.verbose:
            print(f"Done simulating using {self.name} in {self.time_elapsed:.2e} seconds.")

    def simulate(self):
        self._start_simulation()
        result = self._simulate()
        self._end_simulation()
        return result

    def _simulate(self):
        raise NotImplementedError


class SimuPointProcess(Simu):
    """Base class for point-process simulations."""

    def __init__(
        self,
        end_time: float | None = None,
        max_jumps: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ):
        super().__init__(seed=seed, verbose=verbose)
        self._time = 0.0
        self._end_time: float | None = None
        self.end_time = end_time
        self.max_jumps = max_jumps
        self._timestamps: list[list[float]] = []
        self._intensity_track_step = -1.0
        self._next_track_time = 0.0
        self._tracked_times: list[float] = []
        self._tracked_intensity: list[list[float]] = []
        self._tracked_compensator: list[np.ndarray] = []
        self._threshold_negative_intensity = False

    @property
    def end_time(self) -> float | None:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float | None) -> None:
        if value is None:
            self._end_time = None
            return
        value = float(value)
        if value < self.simulation_time:
            raise ValueError(
                "This point process has already been simulated until time "
                f"{self.simulation_time:f}, you cannot set a smaller end_time ({value:f})"
            )
        self._end_time = value

    @property
    def n_nodes(self) -> int:
        raise NotImplementedError

    @property
    def simulation_time(self) -> float:
        return float(self._time)

    @property
    def n_total_jumps(self) -> int:
        return int(sum(len(ts) for ts in self._timestamps))

    @property
    def timestamps(self) -> list[np.ndarray]:
        return [np.asarray(ts, dtype=float) for ts in self._timestamps]

    @property
    def tracked_intensity(self) -> list[np.ndarray]:
        if not self.is_intensity_tracked():
            raise ValueError("Intensity has not been tracked")
        return [np.asarray(values, dtype=float) for values in self._tracked_intensity]

    @property
    def intensity_tracked_times(self) -> np.ndarray:
        if not self.is_intensity_tracked():
            raise ValueError("Intensity has not been tracked")
        return np.asarray(self._tracked_times, dtype=float)

    @property
    def intensity_track_step(self) -> float:
        return self._intensity_track_step

    @property
    def tracked_compensator(self) -> list[np.ndarray]:
        return self._tracked_compensator

    def _init_storage(self) -> None:
        self._timestamps = [[] for _ in range(self.n_nodes)]
        self._tracked_intensity = [[] for _ in range(self.n_nodes)]
        self._tracked_times = []
        self._next_track_time = 0.0
        self._tracked_compensator = [np.asarray([], dtype=float) for _ in range(self.n_nodes)]

    def reset(self) -> None:
        self._time = 0.0
        self._init_storage()

    def track_intensity(self, intensity_track_step: float = -1.0) -> None:
        self._intensity_track_step = float(intensity_track_step)
        self._tracked_times = []
        self._next_track_time = 0.0
        self._tracked_intensity = [[] for _ in range(self.n_nodes)]

    def is_intensity_tracked(self) -> bool:
        return self._intensity_track_step > 0.0

    def threshold_negative_intensity(self, allow: bool = True) -> None:
        self._threshold_negative_intensity = bool(allow)

    def _intensity_at(self, t: float, include_current_jumps: bool = False) -> np.ndarray:
        del include_current_jumps
        raise NotImplementedError

    def _total_intensity_bound(self, t: float, include_current_jumps: bool = True) -> float:
        intensity = self._handle_negative_intensity(
            self._intensity_at(t, include_current_jumps=include_current_jumps)
        )
        return float(np.sum(np.maximum(intensity, 0.0)))

    def _record_intensity(self, t: float, include_current_jumps: bool = True) -> None:
        if not self.is_intensity_tracked():
            return
        values = self._handle_negative_intensity(
            self._intensity_at(t, include_current_jumps=include_current_jumps)
        )
        self._tracked_times.append(float(t))
        for i, value in enumerate(values):
            self._tracked_intensity[i].append(float(value))

    def _record_regular_intensity_until(self, stop_time: float) -> None:
        if not self.is_intensity_tracked():
            return
        while self._next_track_time + self._intensity_track_step < stop_time:
            self._next_track_time += self._intensity_track_step
            self._record_intensity(self._next_track_time)

    def _handle_negative_intensity(self, intensity: np.ndarray) -> np.ndarray:
        if np.any(intensity < 0):
            if self._threshold_negative_intensity:
                return np.maximum(intensity, 0.0)
            raise RuntimeError(
                "Simulation stopped because intensity went negative "
                "(you could call ``threshold_negative_intensity`` to allow it)"
            )
        return intensity

    def _simulate(self):
        if self.end_time is None and self.max_jumps is None:
            raise ValueError("Either end_time or max_jumps must be set")
        if self.end_time is not None and float(self.end_time) == self.simulation_time:
            warnings.warn(
                f"This process has already been simulated until time {float(self.end_time):f}",
                UserWarning,
                stacklevel=2,
            )

        end_time = math.inf if self.end_time is None else float(self.end_time)
        max_jumps = math.inf if self.max_jumps is None else int(self.max_jumps)
        if end_time < self._time:
            raise ValueError(
                "This point process has already been simulated until time "
                f"{self._time:f}, you cannot simulate it until {end_time:f}"
            )
        if self._time == 0.0 and self.n_total_jumps == 0 and not self._tracked_times:
            self._record_intensity(0.0)

        while self._time < end_time and self.n_total_jumps < max_jumps:
            bound = self._total_intensity_bound(self._time, include_current_jumps=True)
            if bound <= 0 or not np.isfinite(bound):
                if math.isfinite(end_time):
                    self._record_regular_intensity_until(end_time)
                    self._time = end_time
                break

            candidate_time = self._time + float(self._rng.exponential(1.0 / bound))
            self._record_regular_intensity_until(min(candidate_time, end_time))

            if candidate_time >= end_time:
                self._time = end_time
                break

            self._time = candidate_time
            intensity = self._handle_negative_intensity(
                self._intensity_at(self._time, include_current_jumps=False)
            )
            total_intensity = float(np.sum(intensity))
            if total_intensity <= 0 or self._rng.uniform() * bound > total_intensity:
                continue

            threshold = self._rng.uniform() * total_intensity
            cumulative = np.cumsum(intensity)
            node = int(np.searchsorted(cumulative, threshold, side="right"))
            node = min(node, self.n_nodes - 1)
            self._timestamps[node].append(self._time)
            self._record_intensity(self._time, include_current_jumps=True)

        return self

    def set_timestamps(self, timestamps: list[Any], end_time: float | None = None):
        if len(timestamps) != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} timestamp arrays")
        arrays = []
        for node, ts in enumerate(timestamps):
            arr = np.asarray(ts, dtype=float)
            if arr.ndim != 1:
                raise ValueError("timestamps must be one-dimensional")
            if arr.size and np.any(np.diff(arr) < 0):
                raise ValueError(f"timestamps for node {node} must be sorted")
            if arr.size and arr[0] < 0:
                raise ValueError("timestamps must be non-negative")
            arrays.append(arr)
        if end_time is None:
            end_time = max((float(arr[-1]) for arr in arrays if arr.size), default=0.0)
        self.end_time = float(end_time)
        latest = max((float(arr[-1]) for arr in arrays if arr.size), default=0.0)
        if self.end_time < latest:
            raise ValueError(
                f"end_time={self.end_time} is before latest timestamp {latest}"
            )

        events = sorted(
            (float(t), node) for node, arr in enumerate(arrays) for t in arr
        )
        self.reset()
        for jump_time, node in events:
            self._record_regular_intensity_until(jump_time)
            self._time = jump_time
            self._record_intensity(jump_time, include_current_jumps=True)
            self._timestamps[node].append(jump_time)
        self._record_regular_intensity_until(float(self.end_time))
        self._time = float(self.end_time)
        return self

    def store_compensator_values(self):
        self._tracked_compensator = []
        for node, timestamps in enumerate(self.timestamps):
            values = [self._evaluate_compensator(node, float(t)) for t in timestamps]
            self._tracked_compensator.append(np.asarray(values, dtype=float))
        return self

    def _evaluate_compensator(self, node: int, t: float) -> float:
        intensity = self._intensity_at
        xs = np.linspace(0.0, t, max(16, int(256 * max(t, 1.0))))
        ys = np.asarray([intensity(float(x))[node] for x in xs], dtype=float)
        return float(np.trapezoid(np.maximum(ys, 0.0), xs))


class SimuPoissonProcess(SimuPointProcess):
    """Homogeneous Poisson process."""

    def __init__(
        self,
        intensities: float | Any,
        end_time: float | None = None,
        max_jumps: int | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ):
        self._intensities_is_scalar = np.isscalar(intensities)
        self._intensities = np.atleast_1d(np.asarray(intensities, dtype=float))
        if np.any(self._intensities < 0):
            raise ValueError("Poisson intensities must be non-negative")
        super().__init__(end_time=end_time, max_jumps=max_jumps, seed=seed, verbose=verbose)
        self._init_storage()

    @property
    def n_nodes(self) -> int:
        return int(self._intensities.size)

    @property
    def intensities(self) -> float | np.ndarray:
        if self._intensities_is_scalar:
            return float(self._intensities[0])
        return self._intensities.copy()

    def _intensity_at(self, t: float, include_current_jumps: bool = False) -> np.ndarray:
        del t
        del include_current_jumps
        return self._intensities.copy()

    def _evaluate_compensator(self, node: int, t: float) -> float:
        return float(self._intensities[node] * t)


class SimuInhomogeneousPoisson(SimuPointProcess):
    """Inhomogeneous Poisson process driven by time functions."""

    def __init__(
        self,
        intensities_functions: list[TimeFunction],
        end_time: float | None = None,
        max_jumps: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ):
        if len(intensities_functions) == 0:
            raise ValueError("at least one intensity function is required")
        self.intensities_functions = list(intensities_functions)
        super().__init__(end_time=end_time, max_jumps=max_jumps, seed=seed, verbose=verbose)
        self._init_storage()

    @property
    def n_nodes(self) -> int:
        return len(self.intensities_functions)

    def intensity_value(self, node: int, times: Any):
        return self.intensities_functions[node].value(times)

    def _intensity_at(self, t: float, include_current_jumps: bool = False) -> np.ndarray:
        del include_current_jumps
        return np.asarray([fn.value(t) for fn in self.intensities_functions], dtype=float)

    def _total_intensity_bound(self, t: float, include_current_jumps: bool = True) -> float:
        del include_current_jumps
        return float(sum(fn.future_bound(t) for fn in self.intensities_functions))

    def _evaluate_compensator(self, node: int, t: float) -> float:
        return float(self.intensities_functions[node].primitive(t))


class SimuHawkes(SimuPointProcess):
    """Hawkes process simulation by thinning."""

    def __init__(
        self,
        kernels: Any | None = None,
        baseline: Any | None = None,
        n_nodes: int | None = None,
        end_time: float | None = None,
        period_length: float | None = None,
        max_jumps: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
        force_simulation: bool = False,
    ):
        self.force_simulation = force_simulation
        self.period_length = period_length
        self._kernel_0 = HawkesKernel0()
        kernels_array, baseline_array, inferred_nodes = self._coerce_parameters(
            kernels, baseline, n_nodes
        )
        self._n_nodes = inferred_nodes
        super().__init__(end_time=end_time, max_jumps=max_jumps, seed=seed, verbose=verbose)
        self.kernels = kernels_array
        self.baseline = baseline_array
        self._init_storage()

    @staticmethod
    def _coerce_parameters(kernels: Any, baseline: Any, n_nodes: int | None):
        if kernels is not None:
            kernels = np.asarray(kernels, dtype=object)
        if baseline is not None:
            baseline = np.asarray(baseline, dtype=object if _contains_time_function(baseline) else float)

        if n_nodes is not None and (kernels is not None or baseline is not None):
            raise ValueError("n_nodes is inferred when kernels or baseline are provided")
        if n_nodes is None:
            if baseline is not None:
                n_nodes = int(baseline.shape[0])
            elif kernels is not None:
                n_nodes = int(kernels.shape[0])
            else:
                raise ValueError("n_nodes must be provided if kernels and baseline are None")
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")

        if kernels is None:
            zero_kernel = HawkesKernel0()
            kernels = np.empty((n_nodes, n_nodes), dtype=object)
            kernels[:, :] = zero_kernel
        if kernels.shape != (n_nodes, n_nodes):
            raise ValueError(f"kernels shape should be {(n_nodes, n_nodes)}")
        zero_kernel = HawkesKernel0()
        clean_kernels = np.empty((n_nodes, n_nodes), dtype=object)
        for i, j in product(range(n_nodes), range(n_nodes)):
            value = kernels[i, j]
            if isinstance(value, HawkesKernel):
                clean_kernels[i, j] = value
            elif value == 0:
                clean_kernels[i, j] = zero_kernel
            else:
                clean_kernels[i, j] = value
            if not isinstance(clean_kernels[i, j], HawkesKernel):
                raise ValueError("kernel entries must be HawkesKernel objects or 0")

        if baseline is None:
            baseline = np.zeros(n_nodes, dtype=float)
        if baseline.shape[0] != n_nodes:
            raise ValueError("baseline length does not match n_nodes")
        if baseline.ndim > 2:
            raise ValueError("baseline must have at most two dimensions")
        return clean_kernels, baseline, n_nodes

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def check_parameters_coherence(self, kernels=None, baseline=None, n_nodes=None) -> None:
        set_kernels = kernels is not None
        set_baseline = baseline is not None
        set_n_nodes = n_nodes is not None
        if set_n_nodes and (set_kernels or set_baseline):
            raise ValueError(
                "n_nodes will be automatically calculated if baseline or kernels is set"
            )
        if not set_n_nodes and not set_kernels and not set_baseline:
            raise ValueError("n_nodes must be given if neither kernels, nor baseline are given")
        if set_kernels and set_baseline and len(kernels) != len(baseline):
            raise ValueError(
                "kernels and baseline have different length. "
                f"kernels has length {len(kernels)}, whereas baseline has length {len(baseline)}."
            )

    def set_kernel(self, i: int, j: int, kernel: HawkesKernel | float | int) -> None:
        self.kernels[i, j] = HawkesKernel0() if kernel == 0 else kernel

    def set_baseline(self, i: int, baseline: Any) -> None:
        self.baseline[i] = baseline

    def _baseline_value(self, i: int, t: float) -> float:
        value = self.baseline[i]
        if isinstance(value, TimeFunction):
            return float(value.value(t))
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        if self.period_length is None:
            raise ValueError("period_length is required for piecewise baseline arrays")
        n_intervals = arr.size
        idx = int(math.floor(((t % self.period_length) / self.period_length) * n_intervals))
        return float(arr[min(idx, n_intervals - 1)])

    def _baseline_bound(self, i: int, t: float) -> float:
        value = self.baseline[i]
        if isinstance(value, TimeFunction):
            return float(value.future_bound(t))
        arr = np.asarray(value, dtype=float)
        return float(max(np.max(arr), 0.0))

    def get_baseline_values(self, i: int, t_values: Any) -> np.ndarray:
        arr = np.asarray(t_values, dtype=float)
        return np.vectorize(lambda t: self._baseline_value(i, float(t)), otypes=[float])(arr)

    def _intensity_at(self, t: float, include_current_jumps: bool = False) -> np.ndarray:
        values = np.zeros(self.n_nodes, dtype=float)
        timestamps = self.timestamps
        for i in range(self.n_nodes):
            values[i] = self._baseline_value(i, t)
            for j in range(self.n_nodes):
                values[i] += _kernel_convolution(
                    self.kernels[i, j],
                    t,
                    timestamps[j],
                    include_current_jumps=include_current_jumps,
                )
        return values

    def _total_intensity_bound(self, t: float, include_current_jumps: bool = True) -> float:
        timestamps = self.timestamps
        total = 0.0
        for i in range(self.n_nodes):
            total += self._baseline_bound(i, t)
            for j in range(self.n_nodes):
                kernel = self.kernels[i, j]
                if kernel.is_zero():
                    continue
                if isinstance(kernel, (HawkesKernelExp, HawkesKernelSumExp)):
                    total += max(
                        _kernel_convolution(
                            kernel,
                            t,
                            timestamps[j],
                            include_current_jumps=include_current_jumps,
                        ),
                        0.0,
                    )
                else:
                    for tj in timestamps[j]:
                        if tj < t or (include_current_jumps and tj <= t):
                            total += kernel.future_bound(t - tj)
        return float(max(total, 0.0))

    def _simulate(self):
        if self.baseline.dtype != object and np.linalg.norm(self.baseline.astype(float)) == 0:
            warnings.warn("Baselines have not been set, so this Hawkes process may not jump")
        radius = self.spectral_radius()
        if not self.force_simulation and self.max_jumps is None and radius >= 1.0:
            raise ValueError(
                "Simulation not launched as this Hawkes process is not stable "
                f"(spectral radius of {radius:.2g}). You can use force_simulation "
                "parameter if you really want to simulate it"
            )
        return super()._simulate()

    def spectral_radius(self) -> float:
        norms = np.empty((self.n_nodes, self.n_nodes), dtype=float)
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            norms[i, j] = self.kernels[i, j].get_norm()
        if norms.size == 1:
            return float(norms[0, 0])
        eigvals = np.linalg.eigvals(norms)
        eigvals = np.real_if_close(eigvals)
        if np.isrealobj(eigvals):
            return float(np.max(eigvals.real))
        return float(np.max(np.abs(eigvals)))

    def mean_intensity(self) -> np.ndarray:
        norms = np.empty((self.n_nodes, self.n_nodes), dtype=float)
        base = np.empty(self.n_nodes, dtype=float)
        for i in range(self.n_nodes):
            base[i] = self._baseline_value(i, 0.0)
            for j in range(self.n_nodes):
                norms[i, j] = self.kernels[i, j].get_norm()
        return np.linalg.solve(np.eye(self.n_nodes) - norms, base)

    def _evaluate_compensator(self, node: int, t: float) -> float:
        value = self._baseline_primitive(node, t)
        timestamps = self.timestamps
        for j in range(self.n_nodes):
            value += self.kernels[node, j].get_primitive_convolution(t, timestamps[j])
        return float(value)

    def _baseline_primitive(self, node: int, t: float) -> float:
        value = self.baseline[node]
        if isinstance(value, TimeFunction):
            return float(value.primitive(t))
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * t
        if self.period_length is None:
            raise ValueError("period_length is required for piecewise baseline arrays")
        dt = self.period_length / arr.size
        full_periods = int(t // self.period_length)
        rem = t - full_periods * self.period_length
        total = full_periods * float(np.sum(arr) * dt)
        for idx, baseline in enumerate(arr):
            start = idx * dt
            stop = min((idx + 1) * dt, rem)
            if stop > start:
                total += float(baseline) * (stop - start)
        return float(total)


class SimuHawkesExpKernels(SimuHawkes):
    """Hawkes simulation with exponential kernels."""

    def __init__(
        self,
        adjacency: Any,
        decays: float | Any,
        baseline: Any | None = None,
        end_time: float | None = None,
        period_length: float | None = None,
        max_jumps: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
        force_simulation: bool = False,
    ):
        self.adjacency = np.asarray(adjacency, dtype=float)
        if self.adjacency.ndim != 2 or self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency matrix must be square")
        self.decays = decays if isinstance(decays, (int, float)) else np.asarray(decays, dtype=float)
        if not isinstance(self.decays, (int, float)) and self.decays.shape != self.adjacency.shape:
            raise ValueError("decays must be scalar or have adjacency shape")
        super().__init__(
            kernels=self._build_exp_kernels(),
            baseline=baseline,
            end_time=end_time,
            period_length=period_length,
            max_jumps=max_jumps,
            seed=seed,
            verbose=verbose,
            force_simulation=force_simulation,
        )

    def _build_exp_kernels(self) -> np.ndarray:
        n_nodes = self.adjacency.shape[0]
        kernels = np.empty((n_nodes, n_nodes), dtype=object)
        zero_kernel = HawkesKernel0()
        for i, j in product(range(n_nodes), range(n_nodes)):
            intensity = float(self.adjacency[i, j])
            decay = float(self.decays if isinstance(self.decays, (int, float)) else self.decays[i, j])
            kernels[i, j] = zero_kernel if intensity == 0 else HawkesKernelExp(intensity, decay)
        return kernels

    def adjust_spectral_radius(self, spectral_radius: float) -> None:
        original = self.spectral_radius()
        if original == 0:
            raise ValueError("cannot adjust spectral radius of a zero kernel matrix")
        self.adjacency = self.adjacency * float(spectral_radius) / original
        self.kernels = self._build_exp_kernels()


class SimuHawkesSumExpKernels(SimuHawkes):
    """Hawkes simulation with sum-exponential kernels."""

    def __init__(
        self,
        adjacency: Any,
        decays: Any,
        baseline: Any | None = None,
        end_time: float | None = None,
        period_length: float | None = None,
        max_jumps: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
        force_simulation: bool = False,
    ):
        self.adjacency = np.asarray(adjacency, dtype=float)
        self.decays = np.asarray(decays, dtype=float)
        if self.adjacency.ndim != 3:
            raise ValueError("adjacency must have shape (n_nodes, n_nodes, n_decays)")
        if self.decays.ndim != 1 or self.adjacency.shape[2] != self.decays.size:
            raise ValueError("decays length must match adjacency third dimension")
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("adjacency first two dimensions must be square")
        super().__init__(
            kernels=self._build_sumexp_kernels(),
            baseline=baseline,
            end_time=end_time,
            period_length=period_length,
            max_jumps=max_jumps,
            seed=seed,
            verbose=verbose,
            force_simulation=force_simulation,
        )

    @property
    def n_decays(self) -> int:
        return int(self.decays.size)

    def _build_sumexp_kernels(self) -> np.ndarray:
        n_nodes = self.adjacency.shape[0]
        kernels = np.empty((n_nodes, n_nodes), dtype=object)
        zero_kernel = HawkesKernel0()
        for i, j in product(range(n_nodes), range(n_nodes)):
            intensities = self.adjacency[i, j, :]
            kernels[i, j] = (
                zero_kernel
                if np.allclose(intensities, 0.0)
                else HawkesKernelSumExp(intensities, self.decays)
            )
        return kernels

    def adjust_spectral_radius(self, spectral_radius: float) -> None:
        original = self.spectral_radius()
        if original == 0:
            raise ValueError("cannot adjust spectral radius of a zero kernel matrix")
        self.adjacency = self.adjacency * float(spectral_radius) / original
        self.kernels = self._build_sumexp_kernels()


class SimuHawkesMulti(Simu):
    """Run repeated Hawkes simulations."""

    def __init__(self, hawkes_simu: SimuHawkes, n_simulations: int, n_threads: int = 1):
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        self.hawkes_simu = hawkes_simu
        self.n_simulations = int(n_simulations)
        self.n_threads = int(n_threads) if n_threads > 0 else multiprocessing.cpu_count()
        self._simulations = [copy.deepcopy(hawkes_simu) for _ in range(n_simulations)]
        super().__init__(seed=hawkes_simu.seed, verbose=hawkes_simu.verbose)

    @property
    def seed(self) -> int | None:
        return self.hawkes_simu.seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        if hasattr(self, "hawkes_simu") and hasattr(self, "_simulations"):
            self.reseed_simulations(value)
        else:
            Simu.seed.fset(self, value)

    def reseed_simulations(self, seed: int | None) -> None:
        seed = None if seed is None else int(seed)
        self._seed = seed
        rng_seed = None if seed is None or seed < 0 else seed
        self._rng = np.random.default_rng(rng_seed)
        self.hawkes_simu.seed = seed
        if seed is None or seed < 0:
            seeds: list[int | None] = [None] * self.n_simulations
        else:
            seeds = [
                int(sim_seed)
                for sim_seed in self._rng.integers(
                    0, 2**31 - 1, size=self.n_simulations, dtype=np.int64
                )
            ]
        for simulation, sim_seed in zip(self._simulations, seeds):
            simulation.seed = sim_seed

    @property
    def n_total_jumps(self) -> list[int]:
        return [simu.n_total_jumps for simu in self._simulations]

    @property
    def timestamps(self) -> list[list[np.ndarray]]:
        return [simu.timestamps for simu in self._simulations]

    @property
    def end_time(self) -> list[float | None]:
        return [simu.end_time for simu in self._simulations]

    @end_time.setter
    def end_time(self, end_times: list[float]):
        if len(end_times) != self.n_simulations:
            raise ValueError(f"end_time must have length {self.n_simulations}")
        for simu, end_time in zip(self._simulations, end_times):
            simu.end_time = end_time

    @property
    def max_jumps(self) -> list[int | None]:
        return [simu.max_jumps for simu in self._simulations]

    @property
    def simulation_time(self) -> list[float]:
        return [simu.simulation_time for simu in self._simulations]

    @property
    def n_nodes(self) -> list[int]:
        return [simu.n_nodes for simu in self._simulations]

    @property
    def spectral_radius(self) -> list[float]:
        return [simu.spectral_radius() for simu in self._simulations]

    @property
    def mean_intensity(self) -> list[np.ndarray]:
        return [simu.mean_intensity() for simu in self._simulations]

    def get_single_simulation(self, i: int) -> SimuHawkes:
        return self._simulations[i]

    def _simulate(self):
        executor_cls = ThreadPoolExecutor
        try:
            with executor_cls(max_workers=self.n_threads) as executor:
                self._simulations = list(executor.map(_simulate_single, self._simulations))
        except Exception:
            with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                self._simulations = list(executor.map(_simulate_single, self._simulations))
        return self


def _simulate_single(simulation: SimuHawkes) -> SimuHawkes:
    simulation.simulate()
    return simulation


def _kernel_convolution(
    kernel: HawkesKernel,
    time: float,
    timestamps: np.ndarray,
    include_current_jumps: bool = False,
) -> float:
    if isinstance(kernel, HawkesKernelExp):
        return exp_kernel_convolution(
            time,
            timestamps,
            kernel.intensity,
            kernel.decay,
            include_current=include_current_jumps,
        )
    if isinstance(kernel, HawkesKernelSumExp):
        return sumexp_kernel_convolution(
            time,
            timestamps,
            kernel.intensities,
            kernel.decays,
            include_current=include_current_jumps,
        )
    if include_current_jumps:
        return float(sum(kernel.get_value(time - tk) for tk in timestamps if tk <= time))
    return kernel.get_convolution(time, timestamps)


def _contains_time_function(value: Any) -> bool:
    if isinstance(value, TimeFunction):
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return any(isinstance(v, TimeFunction) for v in np.asarray(value, dtype=object).flat)
    return False
