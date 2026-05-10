"""Hawkes inference and learner classes."""

from __future__ import annotations

import math
import warnings
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import OptimizeResult
import scipy.linalg

from our_hawkes.base import BaseEstimator, History, normalize_events, relative_distance

from .models import (
    ModelHawkesExpKernLeastSq,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik,
)
from .simulation import SimuHawkesExpKernels, SimuHawkesSumExpKernels
from .solvers import (
    ProxElasticNet,
    ProxL1,
    ProxL2Sq,
    ProxNuclear,
    ProxPositive,
    _penalty_value as _solver_penalty_value,
    optimize_positive_coeffs,
)


class _LearnerBase(BaseEstimator):
    def __init__(
        self,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        n_threads: int = 1,
    ):
        self.tol = tol
        self.max_iter = int(max_iter)
        self.verbose = verbose
        self.print_every = print_every
        self.record_every = record_every
        self.n_threads = n_threads
        self.history = History()
        self._fitted = False
        self.data = None
        self._end_times = None

    def _should_record_iter(self, i: int) -> bool:
        return i == 0 or (i + 1) % max(self.record_every, 1) == 0

    def _record(self, **kwargs):
        self.history.append(**kwargs)
        if self.verbose and kwargs.get("force", False):
            printable = ", ".join(
                f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in kwargs.items()
                if key != "force"
            )
            print(printable)

    def _set_data(self, events, end_times=None):
        self.data, self._end_times, self._n_nodes = normalize_events(events, end_times)
        self._fitted = True

    @property
    def n_nodes(self) -> int:
        if not self._fitted:
            raise ValueError("fit must be called first")
        return int(self._n_nodes)

    @property
    def n_realizations(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)

    @property
    def n_jumps(self) -> int:
        if self.data is None:
            return 0
        return int(sum(ts.size for realization in self.data for ts in realization))

    @property
    def end_times(self):
        return None if self._end_times is None else self._end_times.copy()


class _ParametricHawkesLearner(_LearnerBase):
    _allowed_solvers = {"gd", "agd", "bfgs", "svrg", "sgd", "lbfgs", "lbfgsb", "l-bfgs", "l-bfgs-b"}
    _allowed_penalties = {"none", "l1", "l2", "elasticnet", "nuclear"}
    _prox_classes = {
        "none": ProxPositive,
        "l1": ProxL1,
        "l2": ProxL2Sq,
        "elasticnet": ProxElasticNet,
        "nuclear": ProxNuclear,
    }

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1e3,
        solver: str = "agd",
        step: float | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        elastic_net_ratio: float = 0.95,
        random_state: int | None = None,
        warm_start: bool = False,
    ):
        super().__init__(
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            print_every=print_every,
            record_every=record_every,
        )
        penalty = str(penalty).lower()
        solver = str(solver).lower()
        if penalty not in self._allowed_penalties:
            raise ValueError(f"unknown penalty {penalty!r}")
        if solver not in self._allowed_solvers:
            raise ValueError(f"unknown solver {solver!r}")
        if C is None or C <= 0:
            raise ValueError("C must be positive")
        if random_state is not None and random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not 0.0 <= elastic_net_ratio <= 1.0:
            raise ValueError("elastic_net_ratio must be between 0 and 1")
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.step = step
        self.elastic_net_ratio = elastic_net_ratio
        self.random_state = random_state
        self.warm_start = bool(warm_start)
        self.coeffs = None
        self.events = None
        self._model_obj = None
        self._solver_obj = self._make_solver_obj()
        self._prox_obj = self._make_prox_obj()
        self.history.print_order = ["n_iter", "obj", "rel_obj", "rel_coeffs"]

    def _fit_model(self, model, events, end_times=None, start=None):
        self._set_data(events, end_times)
        self.events = events
        model.fit(events, end_times=end_times)
        self._model_obj = model
        start_coeffs = np.maximum(self._coerce_start(model, start), 0.0)
        self.history.clear()
        self._sync_solver_obj()
        self._prox_obj = self._make_prox_obj()
        if self.penalty == "nuclear":
            self._prox_obj.n_rows = self.n_nodes
        self._record_optimizer_state(model, 0, start_coeffs, None)
        callback_state = {"n_iter": 0, "last_x": start_coeffs.copy(), "last_obj": self._objective_value(model, start_coeffs)}

        def callback(x):
            callback_state["n_iter"] += 1
            n_iter = callback_state["n_iter"]
            if self._should_record_iter(n_iter - 1):
                self._record_optimizer_state(
                    model,
                    n_iter,
                    np.asarray(x, dtype=float),
                    callback_state,
                )

        def result_callback(result: OptimizeResult):
            n_iter = int(getattr(result, "nit", callback_state["n_iter"]) or 0)
            x = np.asarray(getattr(result, "x", start_coeffs), dtype=float)
            if len(self.history) == 0 or self.history[-1].get("n_iter") != n_iter:
                self._record_optimizer_state(model, n_iter, x, callback_state)

        self.coeffs = optimize_positive_coeffs(
            model,
            start_coeffs,
            penalty=self.penalty,
            C=self.C,
            elastic_net_ratio=self.elastic_net_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            jac=self.penalty != "nuclear",
            callback=callback,
            result_callback=result_callback,
        )
        self._fitted = True
        return self

    def _coerce_start(self, model, start):
        if start is None and self.warm_start and self.coeffs is not None:
            previous = np.asarray(self.coeffs, dtype=float)
            if previous.shape == (model.n_coeffs,):
                return previous.copy()
        if start is None:
            return np.ones(model.n_coeffs, dtype=float)
        if isinstance(start, (int, float, np.floating)):
            return np.full(model.n_coeffs, float(start), dtype=float)
        start_coeffs = np.asarray(start, dtype=float).copy()
        if start_coeffs.shape != (model.n_coeffs,):
            raise ValueError(f"start has shape {start_coeffs.shape}, expected {(model.n_coeffs,)}")
        return start_coeffs

    def _make_solver_obj(self):
        return SimpleNamespace(
            name=self.solver,
            method="L-BFGS-B",
            step=self.step,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            print_every=self.print_every,
            record_every=self.record_every,
            seed=self.random_state,
        )

    def _sync_solver_obj(self):
        self._solver_obj.name = self.solver
        self._solver_obj.step = self.step
        self._solver_obj.tol = self.tol
        self._solver_obj.max_iter = self.max_iter
        self._solver_obj.verbose = self.verbose
        self._solver_obj.print_every = self.print_every
        self._solver_obj.record_every = self.record_every
        self._solver_obj.seed = self.random_state

    def _make_prox_obj(self):
        cls = self._prox_classes[self.penalty]
        if self.penalty == "none":
            return cls(0.0)
        if self.penalty == "elasticnet":
            return cls(1.0 / self.C, self.elastic_net_ratio)
        if self.penalty == "nuclear":
            return cls(1.0 / self.C)
        return cls(1.0 / self.C)

    def _objective_value(self, model, coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        strength = 0.0 if self.penalty == "none" else 1.0 / self.C
        value = model.loss(coeffs) + _solver_penalty_value(
            coeffs,
            self.penalty,
            strength,
            self.elastic_net_ratio,
            model,
        )
        return float(value) if np.isfinite(value) else 1e300

    def _record_optimizer_state(self, model, n_iter, coeffs, state):
        obj = self._objective_value(model, coeffs)
        if state is None:
            rel_obj = 0.0
            rel_coeffs = 0.0
        else:
            prev_obj = float(state["last_obj"])
            prev_x = np.asarray(state["last_x"], dtype=float)
            rel_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1.0)
            rel_coeffs = relative_distance(np.asarray(coeffs, dtype=float), prev_x)
            state["last_obj"] = obj
            state["last_x"] = np.asarray(coeffs, dtype=float).copy()
        self._record(n_iter=int(n_iter), obj=obj, rel_obj=rel_obj, rel_coeffs=rel_coeffs)

    @property
    def baseline(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return self.coeffs[: self.n_nodes]

    def score(self, events=None, end_times=None, coeffs=None):
        if events is None and not self._fitted:
            raise ValueError("You must either call `fit` before `score` or provide events")
        if coeffs is None:
            coeffs = self.coeffs
        if coeffs is None:
            raise ValueError("coeffs must be provided when scoring before fit")
        if events is None:
            if end_times is None:
                model = self._model_obj
            else:
                model = self._construct_model_obj()
                model.fit(self.data, end_times=end_times)
        else:
            model = self._construct_model_obj()
            model.fit(events, end_times=end_times)
        return -model.loss(np.asarray(coeffs, dtype=float))

    def estimated_intensity(self, events, intensity_track_step, end_time=None):
        if not self._fitted:
            raise ValueError("fit must be called first")
        if end_time is None:
            end_time = max((float(ts[-1]) for ts in events if len(ts)), default=0.0)
        simu = self._corresponding_simu()
        if intensity_track_step is None:
            intensity_track_step = max(float(end_time), 1.0) / 1000.0
        simu.track_intensity(intensity_track_step)
        simu.set_timestamps(events, end_time=end_time)
        return simu.tracked_intensity, simu.intensity_tracked_times

    def get_kernel_supports(self):
        simu = self._corresponding_simu()
        return np.vectorize(lambda kernel: kernel.get_plot_support())(simu.kernels)

    def get_kernel_values(self, i, j, abscissa_array):
        return self._corresponding_simu().kernels[i, j].get_values(abscissa_array)

    def get_kernel_norms(self):
        simu = self._corresponding_simu()
        return np.vectorize(lambda kernel: kernel.get_norm())(simu.kernels)


class HawkesExpKern(_ParametricHawkesLearner):
    """Fixed-decay exponential Hawkes learner."""

    def __init__(
        self,
        decays,
        gofit: str = "least-squares",
        penalty: str = "l2",
        C: float = 1e3,
        solver: str = "agd",
        step: float | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        elastic_net_ratio: float = 0.95,
        random_state: int | None = None,
        warm_start: bool = False,
    ):
        if gofit not in {"least-squares", "likelihood"}:
            raise ValueError("gofit must be 'least-squares' or 'likelihood'")
        self.gofit = gofit
        self.decays = decays
        super().__init__(
            penalty=penalty,
            C=C,
            solver=solver,
            step=step,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            print_every=print_every,
            record_every=record_every,
            elastic_net_ratio=elastic_net_ratio,
            random_state=random_state,
            warm_start=warm_start,
        )

    def _construct_model_obj(self):
        if self.gofit == "least-squares":
            return ModelHawkesExpKernLeastSq(self.decays)
        if not isinstance(self.decays, (int, float, np.floating)):
            decays = np.asarray(self.decays, dtype=float)
            if not np.allclose(decays, decays.flat[0]):
                raise NotImplementedError("likelihood gofit requires a shared scalar decay")
            decay = float(decays.flat[0])
        else:
            decay = float(self.decays)
        return ModelHawkesExpKernLogLik(decay)

    def fit(self, events, start=None, end_times=None):
        return self._fit_model(self._construct_model_obj(), events, end_times=end_times, start=start)

    @property
    def adjacency(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return self.coeffs[self.n_nodes :].reshape((self.n_nodes, self.n_nodes))

    def _corresponding_simu(self):
        return SimuHawkesExpKernels(
            adjacency=self.adjacency,
            decays=self.decays,
            baseline=self.baseline,
            force_simulation=True,
            verbose=False,
        )

    def score(self, events=None, end_times=None, baseline=None, adjacency=None):
        coeffs = None
        if baseline is not None or adjacency is not None:
            baseline = self.baseline if baseline is None else baseline
            adjacency = self.adjacency if adjacency is None else adjacency
            coeffs = np.hstack((np.asarray(baseline, dtype=float), np.asarray(adjacency, dtype=float).ravel()))
        return super().score(events=events, end_times=end_times, coeffs=coeffs)


class HawkesSumExpKern(_ParametricHawkesLearner):
    """Fixed-decay sum-exponential Hawkes learner."""

    _allowed_penalties = {"none", "l1", "l2", "elasticnet"}

    def __init__(
        self,
        decays,
        penalty: str = "l2",
        C: float = 1e3,
        n_baselines: int = 1,
        period_length: float | None = None,
        solver: str = "agd",
        step: float | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        elastic_net_ratio: float = 0.95,
        random_state: int | None = None,
        warm_start: bool = False,
    ):
        self.decays = np.asarray(decays, dtype=float)
        self.n_baselines = n_baselines
        self.period_length = period_length
        super().__init__(
            penalty=penalty,
            C=C,
            solver=solver,
            step=step,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            print_every=print_every,
            record_every=record_every,
            elastic_net_ratio=elastic_net_ratio,
            random_state=random_state,
            warm_start=warm_start,
        )

    def _construct_model_obj(self):
        return ModelHawkesSumExpKernLeastSq(
            self.decays, n_baselines=self.n_baselines, period_length=self.period_length
        )

    def fit(self, events, start=None, end_times=None):
        return self._fit_model(self._construct_model_obj(), events, end_times=end_times, start=start)

    @property
    def n_decays(self):
        return int(self.decays.size)

    @property
    def baseline(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        raw = self.coeffs[: self.n_nodes * self.n_baselines]
        if self.n_baselines == 1:
            return raw
        return raw.reshape((self.n_nodes, self.n_baselines))

    @property
    def adjacency(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return self.coeffs[self.n_nodes * self.n_baselines :].reshape(
            (self.n_nodes, self.n_nodes, self.n_decays)
        )

    def _corresponding_simu(self):
        return SimuHawkesSumExpKernels(
            adjacency=self.adjacency,
            decays=self.decays,
            baseline=self.baseline,
            period_length=self.period_length,
            force_simulation=True,
            verbose=False,
        )

    def get_baseline_values(self, i, abscissa_array):
        return self._corresponding_simu().get_baseline_values(i, abscissa_array)

    def score(self, events=None, end_times=None, baseline=None, adjacency=None):
        coeffs = None
        if baseline is not None or adjacency is not None:
            baseline = self.baseline if baseline is None else baseline
            adjacency = self.adjacency if adjacency is None else adjacency
            coeffs = np.hstack((np.asarray(baseline, dtype=float).ravel(), np.asarray(adjacency, dtype=float).ravel()))
        return super().score(events=events, end_times=end_times, coeffs=coeffs)


class HawkesEM(_LearnerBase):
    """Non-parametric EM learner with piecewise constant kernels."""

    def __init__(
        self,
        kernel_support: float | None = None,
        kernel_size: int = 10,
        kernel_discretization: Any | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        print_every: int = 10,
        record_every: int = 10,
        verbose: bool = False,
        n_threads: int = 1,
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every, n_threads)
        if kernel_discretization is not None:
            self._set_kernel_discretization(kernel_discretization)
        elif kernel_support is not None:
            self._set_uniform_kernel_grid(kernel_support, kernel_size)
        else:
            raise ValueError("either kernel_support or kernel_discretization must be provided")
        self.baseline = None
        self.kernel = None
        self.history.print_order = ["n_iter", "rel_baseline", "rel_kernel"]

    def _set_uniform_kernel_grid(self, kernel_support, kernel_size):
        support = float(kernel_support)
        size = int(kernel_size)
        if support <= 0 or size <= 0:
            raise ValueError("kernel_support and kernel_size must be positive")
        self._kernel_support = support
        self._kernel_size = size
        self._kernel_discretization = np.linspace(0.0, support, size + 1)

    def _set_kernel_discretization(self, kernel_discretization):
        discretization = np.asarray(kernel_discretization, dtype=float)
        if discretization.ndim != 1 or discretization.size < 2:
            raise ValueError("kernel_discretization must contain at least two points")
        if np.any(np.diff(discretization) <= 0):
            raise ValueError("kernel_discretization must be strictly increasing")
        if discretization[0] < 0:
            raise ValueError("kernel_discretization must be non-negative")
        if discretization[-1] <= 0:
            raise ValueError("kernel_discretization support must be positive")
        self._kernel_discretization = np.ascontiguousarray(discretization, dtype=float)
        self._kernel_support = float(discretization[-1])
        self._kernel_size = int(discretization.size - 1)

    @property
    def kernel_support(self):
        return self._kernel_support

    @kernel_support.setter
    def kernel_support(self, val):
        self._set_uniform_kernel_grid(val, self.kernel_size)

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, val):
        self._set_uniform_kernel_grid(self.kernel_support, val)

    @property
    def kernel_discretization(self):
        return self._kernel_discretization.copy()

    @kernel_discretization.setter
    def kernel_discretization(self, val):
        self._set_kernel_discretization(val)

    @property
    def kernel_dt(self):
        diffs = np.diff(self._kernel_discretization)
        if np.allclose(diffs, diffs[0]):
            return float(diffs[0])
        return diffs

    @kernel_dt.setter
    def kernel_dt(self, val):
        dt = float(val)
        if dt <= 0:
            raise ValueError("kernel_dt must be positive")
        size = int(np.ceil(self.kernel_support / dt))
        self._set_uniform_kernel_grid(self.kernel_support, max(size, 1))

    def fit(self, events, end_times=None, baseline_start=None, kernel_start=None):
        self._set_data(events, end_times)
        if baseline_start is None:
            self.baseline = np.ones(self.n_nodes, dtype=float)
        else:
            self.baseline = np.asarray(baseline_start, dtype=float).copy()
            if self.baseline.shape != (self.n_nodes,):
                raise ValueError(f"baseline_start has shape {self.baseline.shape}, expected {(self.n_nodes,)}")
        if kernel_start is None:
            self.kernel = 0.1 * np.random.uniform(size=(self.n_nodes, self.n_nodes, self.kernel_size))
        else:
            self.kernel = np.asarray(kernel_start, dtype=float).copy()
        if self.kernel.shape != (self.n_nodes, self.n_nodes, self.kernel_size):
            raise ValueError(
                f"kernel_start has shape {self.kernel.shape}, expected "
                f"{(self.n_nodes, self.n_nodes, self.kernel_size)}"
            )

        for i in range(self.max_iter):
            old_baseline = self.baseline.copy()
            old_kernel = self.kernel.copy()
            self._em_step()
            rel_baseline = relative_distance(self.baseline, old_baseline)
            rel_kernel = relative_distance(self.kernel, old_kernel)
            if self._should_record_iter(i):
                self._record(n_iter=i + 1, rel_baseline=rel_baseline, rel_kernel=rel_kernel)
            if max(rel_baseline, rel_kernel) <= self.tol:
                break
        self._fitted = True
        return self

    def _em_step(self):
        total_time = float(np.sum(self._end_times))
        next_baseline = np.zeros(self.n_nodes, dtype=float)
        next_kernel = np.zeros_like(self.kernel)
        node_counts = np.zeros(self.n_nodes, dtype=float)
        dt = np.diff(self._kernel_discretization)
        for realization in self.data:
            for v in range(self.n_nodes):
                node_counts[v] += realization[v].size
        node_counts = np.maximum(node_counts, 1.0)

        for realization in self.data:
            for u in range(self.n_nodes):
                for t in realization[u]:
                    contributions = np.zeros((self.n_nodes, self.kernel_size), dtype=float)
                    intensity = self.baseline[u]
                    for v in range(self.n_nodes):
                        for tj in realization[v]:
                            if tj >= t:
                                break
                            lag = float(t - tj)
                            if lag >= self.kernel_support:
                                continue
                            m = int(np.searchsorted(self._kernel_discretization, lag) - 1)
                            if 0 <= m < self.kernel_size:
                                value = self.kernel[u, v, m]
                                contributions[v, m] += value
                                intensity += value
                    if intensity <= 0:
                        continue
                    next_baseline[u] += self.baseline[u] / intensity
                    next_kernel[u] += contributions / intensity

        self.baseline = np.maximum(next_baseline / max(total_time, 1e-15), 0.0)
        for u, v, m in product(range(self.n_nodes), range(self.n_nodes), range(self.kernel_size)):
            self.kernel[u, v, m] = next_kernel[u, v, m] / (node_counts[v] * dt[m])
        self.kernel = np.maximum(self.kernel, 0.0)

    @property
    def _flat_kernels(self):
        return self.kernel.reshape((self.n_nodes, self.n_nodes * self.kernel_size))

    def get_kernel_supports(self):
        return np.full((self.n_nodes, self.n_nodes), self.kernel_support)

    def get_kernel_values(self, i, j, abscissa_array):
        x = np.asarray(abscissa_array, dtype=float)
        values = np.zeros_like(x)
        mask = (x > 0) & (x < self.kernel_support)
        idx = np.searchsorted(self._kernel_discretization, x[mask]) - 1
        values[mask] = self.kernel[i, j, idx]
        return values

    def _compute_primitive_kernel_values(self, i, j, abscissa_array):
        primitives = self._get_kernel_primitives()
        x = np.asarray(abscissa_array, dtype=float)
        idx = np.clip(np.searchsorted(self._kernel_discretization, x) - 1, 0, self.kernel_size - 1)
        return primitives[i, j, idx]

    def get_kernel_norms(self):
        return np.einsum("ijk,k->ij", self.kernel, np.diff(self._kernel_discretization))

    def _get_kernel_primitives(self):
        dt = np.diff(self._kernel_discretization)
        return np.cumsum(self.kernel * dt[None, None, :], axis=2)

    def score(self, events=None, end_times=None, baseline=None, kernel=None):
        if events is None and not self._fitted:
            raise ValueError("You must either call `fit` before `score` or provide events")
        if events is None and end_times is not None:
            raise ValueError("events must be provided when end_times is provided")
        if events is None:
            data = self.data
            end_times_arr = self._end_times
            n_nodes = self.n_nodes
        else:
            data, end_times_arr, n_nodes = normalize_events(events, end_times)
        baseline = self.baseline if baseline is None else np.asarray(baseline, dtype=float)
        kernel = self.kernel if kernel is None else np.asarray(kernel, dtype=float)
        self._check_score_parameters(n_nodes, baseline, kernel)
        return _piecewise_loglik(data, end_times_arr, baseline, kernel, self._kernel_discretization)

    def time_changed_interarrival_times(self, events=None, end_times=None, baseline=None, kernel=None):
        if events is None and not self._fitted:
            raise ValueError(
                "You must either call `fit` before `time_changed_interarrival_times` or provide events"
            )
        if events is None and end_times is not None:
            raise ValueError("events must be provided when end_times is provided")
        if events is None:
            data = self.data
            end_times_arr = self._end_times
            n_nodes = self.n_nodes
        else:
            data, end_times_arr, n_nodes = normalize_events(events, end_times)
        baseline = self.baseline if baseline is None else np.asarray(baseline, dtype=float)
        kernel = self.kernel if kernel is None else np.asarray(kernel, dtype=float)
        self._check_score_parameters(n_nodes, baseline, kernel)
        out = []
        for realization, end_time in zip(data, end_times_arr):
            del end_time
            out_r = []
            for u in range(n_nodes):
                vals = [
                    _piecewise_compensator_at(float(t), u, realization, baseline, kernel, self._kernel_discretization)
                    for t in realization[u]
                ]
                out_r.append(np.diff(vals))
            out.append(out_r)
        return out

    def _check_score_parameters(self, n_nodes, baseline, kernel):
        if baseline is None or kernel is None:
            raise ValueError("baseline and kernel must be provided unless the learner has been fitted")
        if baseline.shape != (n_nodes,):
            raise ValueError(f"baseline has shape {baseline.shape}, expected {(n_nodes,)}")
        if kernel.shape != (n_nodes, n_nodes, self.kernel_size):
            raise ValueError(
                f"kernel has shape {kernel.shape}, expected {(n_nodes, n_nodes, self.kernel_size)}"
            )

    def objective(self, coeffs, loss=None):
        del coeffs, loss
        raise NotImplementedError()


class HawkesADM4(_LearnerBase):
    """Compatibility ADM4 learner using exponential likelihood plus penalties."""

    def __init__(
        self,
        decay,
        C: float = 1e3,
        lasso_nuclear_ratio: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-5,
        n_threads: int = 1,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        rho: float = 0.1,
        approx: int = 0,
        em_max_iter: int = 30,
        em_tol: float | None = None,
        warm_start: bool = False,
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every, n_threads)
        if decay <= 0:
            raise ValueError("decay must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.decay = float(decay)
        self._C = None
        self._lasso_nuclear_ratio = None
        self.rho = float(rho)
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.warm_start = bool(warm_start)
        self.baseline = None
        self.adjacency = None
        self._model = None
        self._prox_l1 = ProxL1(0.0)
        self._prox_nuclear = ProxNuclear(0.0)
        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.history.print_order = ["n_iter", "obj", "rel_obj", "rel_coeffs"]

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if value is None or value <= 0:
            raise ValueError("C must be positive")
        self._C = float(value)
        self._update_prox_strengths()

    @property
    def lasso_nuclear_ratio(self):
        return self._lasso_nuclear_ratio

    @lasso_nuclear_ratio.setter
    def lasso_nuclear_ratio(self, value):
        if value < 0 or value > 1:
            raise ValueError("lasso_nuclear_ratio must be between 0 and 1")
        self._lasso_nuclear_ratio = float(value)
        self._update_prox_strengths()

    def _update_prox_strengths(self):
        if self._C is None or self._lasso_nuclear_ratio is None:
            return
        self._prox_l1.strength = self.strength_lasso
        self._prox_nuclear.strength = self.strength_nuclear

    @property
    def strength_lasso(self):
        return self.lasso_nuclear_ratio / self.C

    @property
    def strength_nuclear(self):
        return (1.0 - self.lasso_nuclear_ratio) / self.C

    @property
    def coeffs(self):
        if self.baseline is None or self.adjacency is None:
            return None
        return np.hstack((self.baseline, self.adjacency.ravel()))

    def fit(self, events, end_times=None, baseline_start=None, adjacency_start=None):
        self._set_data(events, end_times)
        model = ModelHawkesExpKernLogLik(self.decay, n_threads=self.n_threads).fit(events, end_times=end_times)
        self._model = model
        self._prox_nuclear.n_rows = self.n_nodes
        start = self._coerce_start(baseline_start, adjacency_start)
        self.history.clear()
        self._record_optimizer_state(model, 0, start, None)
        callback_state = {"n_iter": 0, "last_x": start.copy(), "last_obj": self._objective_value(model, start)}

        def callback(x):
            callback_state["n_iter"] += 1
            n_iter = callback_state["n_iter"]
            if self._should_record_iter(n_iter - 1):
                self._record_optimizer_state(model, n_iter, np.asarray(x, dtype=float), callback_state)

        def result_callback(result: OptimizeResult):
            n_iter = int(getattr(result, "nit", callback_state["n_iter"]) or 0)
            x = np.asarray(getattr(result, "x", start), dtype=float)
            if len(self.history) == 0 or self.history[-1].get("n_iter") != n_iter:
                self._record_optimizer_state(model, n_iter, x, callback_state)

        coeffs = optimize_positive_coeffs(
            model,
            start,
            penalty="none",
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            jac=False,
            callback=callback,
            extra_penalty=lambda x: self._adjacency_penalty_from_coeffs(x),
            result_callback=result_callback,
        )
        self.baseline, self.adjacency = self._unpack_coeffs(coeffs)
        self._fitted = True
        return self

    def _coerce_start(self, baseline_start, adjacency_start):
        if baseline_start is None and adjacency_start is None and self.warm_start and self.coeffs is not None:
            return np.asarray(self.coeffs, dtype=float).copy()
        if baseline_start is None:
            baseline = np.ones(self.n_nodes, dtype=float)
        else:
            baseline = np.asarray(baseline_start, dtype=float).copy()
            if baseline.shape != (self.n_nodes,):
                raise ValueError(f"baseline_start has shape {baseline.shape}, expected {(self.n_nodes,)}")
        if adjacency_start is None:
            adjacency = np.full((self.n_nodes, self.n_nodes), 0.1, dtype=float)
        else:
            adjacency = np.asarray(adjacency_start, dtype=float).copy()
            if adjacency.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    f"adjacency_start has shape {adjacency.shape}, expected {(self.n_nodes, self.n_nodes)}"
                )
        baseline = np.maximum(baseline, 1e-12)
        adjacency = np.maximum(adjacency, 0.0)
        return np.hstack((baseline, adjacency.ravel()))

    def _unpack_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.shape != (self.n_nodes + self.n_nodes * self.n_nodes,):
            raise ValueError("coeffs has wrong shape")
        baseline = coeffs[: self.n_nodes]
        adjacency = coeffs[self.n_nodes :].reshape((self.n_nodes, self.n_nodes))
        return baseline, adjacency

    def _adjacency_penalty_from_coeffs(self, coeffs):
        _, adjacency = self._unpack_coeffs(coeffs)
        return self._prox_l1.value(adjacency.ravel()) + self._prox_nuclear.value(adjacency)

    def _objective_value(self, model, coeffs):
        value = model.loss(np.asarray(coeffs, dtype=float)) + self._adjacency_penalty_from_coeffs(coeffs)
        return float(value) if np.isfinite(value) else 1e300

    def _record_optimizer_state(self, model, n_iter, coeffs, state):
        obj = self._objective_value(model, coeffs)
        if state is None:
            rel_obj = 0.0
            rel_coeffs = 0.0
        else:
            prev_obj = float(state["last_obj"])
            prev_x = np.asarray(state["last_x"], dtype=float)
            rel_obj = abs(obj - prev_obj) / max(abs(prev_obj), 1.0)
            rel_coeffs = relative_distance(np.asarray(coeffs, dtype=float), prev_x)
            state["last_obj"] = obj
            state["last_x"] = np.asarray(coeffs, dtype=float).copy()
        self._record(n_iter=int(n_iter), obj=obj, rel_obj=rel_obj, rel_coeffs=rel_coeffs)

    def objective(self, coeffs, loss=None):
        if not self._fitted and self._model is None:
            raise ValueError("fit must be called first")
        if loss is None:
            loss = self._model.loss(coeffs)
        return float(loss + self._adjacency_penalty_from_coeffs(coeffs))

    def _corresponding_simu(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return SimuHawkesExpKernels(self.adjacency, self.decay, self.baseline, force_simulation=True, verbose=False)

    def get_kernel_supports(self):
        return np.vectorize(lambda kernel: kernel.get_plot_support())(self._corresponding_simu().kernels)

    def get_kernel_values(self, i, j, abscissa_array):
        return self._corresponding_simu().kernels[i, j].get_values(abscissa_array)

    def get_kernel_norms(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return self.adjacency.copy()

    def score(self, events=None, end_times=None, baseline=None, adjacency=None):
        if events is None and not self._fitted:
            raise ValueError("You must either call `fit` before `score` or provide events")
        if baseline is None:
            if self.baseline is None:
                raise ValueError("baseline must be provided when scoring before fit")
            baseline = self.baseline
        if adjacency is None:
            if self.adjacency is None:
                raise ValueError("adjacency must be provided when scoring before fit")
            adjacency = self.adjacency
        coeffs = np.hstack((np.asarray(baseline, dtype=float), np.asarray(adjacency, dtype=float).ravel()))
        if events is None:
            model = self._model if end_times is None else ModelHawkesExpKernLogLik(self.decay).fit(self.data, end_times)
        else:
            model = ModelHawkesExpKernLogLik(self.decay).fit(events, end_times)
        return -model.loss(coeffs)

    def estimated_intensity(self, events, intensity_track_step, end_time=None):
        if not self._fitted:
            raise ValueError("fit must be called first")
        if end_time is None:
            end_time = max((float(ts[-1]) for ts in events if len(ts)), default=0.0)
        if intensity_track_step is None:
            intensity_track_step = max(float(end_time), 1.0) / 1000.0
        simu = self._corresponding_simu()
        simu.track_intensity(intensity_track_step)
        simu.set_timestamps(events, end_time=end_time)
        return simu.tracked_intensity, simu.intensity_tracked_times


class HawkesSumGaussians(_LearnerBase):
    """Sum-of-Gaussians learner initialized from a piecewise EM estimate."""

    def __init__(
        self,
        max_mean_gaussian,
        n_gaussians: int = 5,
        step_size: float = 1e-7,
        C: float = 1e3,
        lasso_grouplasso_ratio: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-5,
        n_threads: int = 1,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        approx: int = 0,
        em_max_iter: int = 30,
        em_tol: float | None = None,
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every, n_threads)
        if max_mean_gaussian <= 0:
            raise ValueError("max_mean_gaussian must be positive")
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.max_mean_gaussian = float(max_mean_gaussian)
        self.n_gaussians = int(n_gaussians)
        self.step_size = step_size
        self._C = 1.0
        self._lasso_grouplasso_ratio = 0.5
        self._strength_lasso = 0.5
        self._strength_grouplasso = 0.5
        self.C = C
        self.lasso_grouplasso_ratio = lasso_grouplasso_ratio
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.baseline = None
        self.amplitudes = None

    def _sync_strengths(self):
        self._strength_lasso = self._lasso_grouplasso_ratio / self._C
        self._strength_grouplasso = (1.0 - self._lasso_grouplasso_ratio) / self._C

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        if val is None or val <= 0:
            raise ValueError("C must be positive")
        self._C = float(val)
        self._sync_strengths()

    @property
    def lasso_grouplasso_ratio(self):
        return self._lasso_grouplasso_ratio

    @lasso_grouplasso_ratio.setter
    def lasso_grouplasso_ratio(self, val):
        if val is None or val < 0 or val > 1:
            raise ValueError("lasso_grouplasso_ratio must be between 0 and 1")
        self._lasso_grouplasso_ratio = float(val)
        self._sync_strengths()

    @property
    def strength_lasso(self):
        return self._strength_lasso

    @strength_lasso.setter
    def strength_lasso(self, val):
        if val < 0:
            raise ValueError("strength_lasso must be non-negative")
        self._strength_lasso = float(val)
        total = self._strength_lasso + self._strength_grouplasso
        if total > 0:
            self._C = 1.0 / total
            self._lasso_grouplasso_ratio = self._strength_lasso / total

    @property
    def strength_grouplasso(self):
        return self._strength_grouplasso

    @strength_grouplasso.setter
    def strength_grouplasso(self, val):
        if val < 0:
            raise ValueError("strength_grouplasso must be non-negative")
        self._strength_grouplasso = float(val)
        total = self._strength_lasso + self._strength_grouplasso
        if total > 0:
            self._C = 1.0 / total
            self._lasso_grouplasso_ratio = self._strength_lasso / total

    @property
    def means_gaussians(self):
        return np.arange(self.n_gaussians) * self.max_mean_gaussian / self.n_gaussians

    @property
    def std_gaussian(self):
        return self.max_mean_gaussian / (self.n_gaussians * np.pi)

    def fit(self, events, end_times=None, baseline_start=None, amplitudes_start=None):
        self._set_data(events, end_times)
        if amplitudes_start is not None:
            self.baseline = (
                np.ones(self.n_nodes)
                if baseline_start is None
                else np.asarray(baseline_start, dtype=float).copy()
            )
            self.amplitudes = np.asarray(amplitudes_start, dtype=float).copy()
            if self.baseline.shape != (self.n_nodes,):
                raise ValueError(f"baseline_start has shape {self.baseline.shape}, expected {(self.n_nodes,)}")
            if self.amplitudes.shape != (self.n_nodes, self.n_nodes, self.n_gaussians):
                raise ValueError(
                    f"amplitudes_start has shape {self.amplitudes.shape}, expected "
                    f"{(self.n_nodes, self.n_nodes, self.n_gaussians)}"
                )
        else:
            em = HawkesEM(
                kernel_support=self.max_mean_gaussian,
                kernel_size=self.n_gaussians,
                max_iter=min(self.max_iter, self.em_max_iter),
                tol=self.tol,
                verbose=False,
            ).fit(events, end_times=end_times, baseline_start=baseline_start)
            self.baseline = em.baseline
            self.amplitudes = em.kernel * (self.max_mean_gaussian / self.n_gaussians)
        self._fitted = True
        return self

    def get_kernel_supports(self):
        return np.full((self.n_nodes, self.n_nodes), self.n_gaussians, dtype=float)

    def get_kernel_values(self, i, j, abscissa_array):
        x = np.asarray(abscissa_array, dtype=float)
        values = np.zeros_like(x)
        sigma = self.std_gaussian
        for m, mean in enumerate(self.means_gaussians):
            values += self.amplitudes[i, j, m] * np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        return values

    def get_kernel_norms(self):
        return np.einsum("ijk->ij", self.amplitudes)

    def objective(self, coeffs, loss=None):
        del coeffs, loss
        raise NotImplementedError()


class HawkesBasisKernels(_LearnerBase):
    """Basis-kernel learner initialized from a piecewise EM estimate."""

    def __init__(
        self,
        kernel_support,
        n_basis=None,
        kernel_size: int = 10,
        tol: float = 1e-5,
        C: float = 1e-1,
        max_iter: int = 100,
        verbose: bool = False,
        print_every: int = 10,
        record_every: int = 10,
        n_threads: int = 1,
        ode_max_iter: int = 100,
        ode_tol: float = 1e-5,
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every, n_threads)
        self._n_basis = 0
        self._C = 1.0
        self._set_uniform_kernel_grid(kernel_support, kernel_size)
        self.n_basis = n_basis
        self.C = C
        self.ode_max_iter = ode_max_iter
        self.ode_tol = ode_tol
        self.baseline = None
        self.amplitudes = None
        self.basis_kernels = None

    def _set_uniform_kernel_grid(self, kernel_support, kernel_size):
        support = float(kernel_support)
        size = int(kernel_size)
        if support <= 0 or size <= 0:
            raise ValueError("kernel_support and kernel_size must be positive")
        self._kernel_support = support
        self._kernel_size = size
        self._kernel_discretization = np.linspace(0.0, support, size + 1)

    @property
    def kernel_support(self):
        return self._kernel_support

    @kernel_support.setter
    def kernel_support(self, val):
        self._set_uniform_kernel_grid(val, self.kernel_size)

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, val):
        self._set_uniform_kernel_grid(self.kernel_support, val)

    @property
    def kernel_discretization(self):
        return self._kernel_discretization.copy()

    @property
    def kernel_dt(self):
        return float(self.kernel_support / self.kernel_size)

    @kernel_dt.setter
    def kernel_dt(self, val):
        dt = float(val)
        if dt <= 0:
            raise ValueError("kernel_dt must be positive")
        size = int(np.ceil(self.kernel_support / dt))
        self._set_uniform_kernel_grid(self.kernel_support, max(size, 1))

    @property
    def n_basis(self):
        return self._n_basis

    @n_basis.setter
    def n_basis(self, val):
        if val is None:
            val = 0
        val = int(val)
        if val < 0:
            raise ValueError("n_basis must be non-negative")
        self._n_basis = val

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        if val is None or val <= 0:
            raise ValueError("C must be positive")
        self._C = float(val)

    def fit(self, events, end_times=None, baseline_start=None, amplitudes_start=None, basis_kernels_start=None):
        self._set_data(events, end_times)
        if self.n_basis in (None, 0):
            self.n_basis = self.n_nodes
        if basis_kernels_start is not None or amplitudes_start is not None:
            self.baseline = np.ones(self.n_nodes) if baseline_start is None else np.asarray(baseline_start, dtype=float).copy()
            if amplitudes_start is None:
                self.amplitudes = np.random.uniform(
                    0.5, 0.9, size=(self.n_nodes, self.n_nodes, self.n_basis)
                )
            else:
                self.amplitudes = np.asarray(amplitudes_start, dtype=float).copy()
            if basis_kernels_start is None:
                self.basis_kernels = 0.1 * np.random.uniform(size=(self.n_basis, self.kernel_size))
            else:
                self.basis_kernels = np.asarray(basis_kernels_start, dtype=float).copy()
            if self.baseline.shape != (self.n_nodes,):
                raise ValueError(f"baseline_start has shape {self.baseline.shape}, expected {(self.n_nodes,)}")
            if self.amplitudes.shape != (self.n_nodes, self.n_nodes, self.n_basis):
                raise ValueError(
                    f"amplitudes_start has shape {self.amplitudes.shape}, expected "
                    f"{(self.n_nodes, self.n_nodes, self.n_basis)}"
                )
            if self.basis_kernels.shape != (self.n_basis, self.kernel_size):
                raise ValueError(
                    f"basis_kernels_start has shape {self.basis_kernels.shape}, expected "
                    f"{(self.n_basis, self.kernel_size)}"
                )
        else:
            em = HawkesEM(self.kernel_support, self.kernel_size, max_iter=self.max_iter, tol=self.tol).fit(
                events, end_times=end_times, baseline_start=baseline_start
            )
            self.baseline = em.baseline
            self.basis_kernels = np.zeros((self.n_basis, self.kernel_size), dtype=float)
            self.amplitudes = np.zeros((self.n_nodes, self.n_nodes, self.n_basis), dtype=float)
            for d in range(self.n_basis):
                basis = np.mean(em.kernel[d % self.n_nodes, :, :], axis=0)
                norm = max(float(np.sum(basis) * self.kernel_dt), 1e-12)
                self.basis_kernels[d] = basis
                for u, v in product(range(self.n_nodes), range(self.n_nodes)):
                    self.amplitudes[u, v, d] = (
                        np.sum(em.kernel[u, v, :] * basis) * self.kernel_dt / norm
                    )
        self._fitted = True
        return self

    def get_kernel_supports(self):
        return np.full((self.n_nodes, self.n_nodes), self.kernel_support)

    def get_kernel_values(self, i, j, abscissa_array):
        x = np.asarray(abscissa_array, dtype=float)
        values = np.zeros_like(x)
        mask = (x > 0) & (x < self.kernel_support)
        idx = np.searchsorted(self._kernel_discretization, x[mask]) - 1
        kernel_ij = np.tensordot(self.amplitudes[i, j, :], self.basis_kernels, axes=(0, 0))
        values[mask] = kernel_ij[idx]
        return values

    def get_kernel_norms(self):
        basis_norms = self.basis_kernels @ np.diff(self._kernel_discretization)
        return np.tensordot(self.amplitudes, basis_norms, axes=(2, 0))

    def objective(self, coeffs, loss=None):
        del coeffs, loss
        raise NotImplementedError()


class HawkesConditionalLaw(_LearnerBase):
    """Empirical conditional-law estimator with tick-compatible outputs."""

    _UNSET = object()

    def __init__(
        self,
        delta_lag: float = 0.1,
        min_lag: float = 1e-4,
        max_lag: float = 40.0,
        n_quad: int = 50,
        max_support: float = 40.0,
        min_support: float = 1e-4,
        quad_method: str = "gauss",
        marked_components=None,
        delayed_component=None,
        delay: float = 0.00001,
        model=None,
        n_threads: int = 1,
        claw_method: str = "lin",
    ):
        super().__init__(n_threads=n_threads)
        if n_quad <= 0:
            raise ValueError("n_quad must be positive")
        if delta_lag <= 0:
            raise ValueError("delta_lag must be positive")
        if max_lag <= 0:
            raise ValueError("max_lag must be positive")
        if min_lag <= 0:
            raise ValueError("min_lag must be positive")
        if max_support <= 0:
            raise ValueError("max_support must be positive")
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        if claw_method not in {"lin", "log"}:
            raise ValueError("claw_method must be one of 'lin' or 'log'")
        if quad_method not in {"gauss", "gauss-", "lin", "log"}:
            raise ValueError("quad_method must be one of 'gauss', 'gauss-', 'lin', or 'log'")
        self.delta_lag = float(delta_lag)
        self.min_lag = float(min_lag)
        self.max_lag = float(max_lag)
        self.n_quad = int(n_quad)
        self.max_support = float(max_support)
        self.min_support = float(min_support)
        self.quad_method = quad_method
        self._marked_components_spec = {} if marked_components is None else dict(marked_components)
        self.marked_components = self._marked_components_spec.copy()
        self.delayed_component = None if delayed_component is None else np.asarray(np.atleast_1d(delayed_component), dtype=int)
        self.delay = float(delay)
        self.claw_method = claw_method
        self.mean_intensity = None
        self.baseline = None
        self.kernels = None
        self.kernels_norms = None
        self.mark_functions = None
        self._marks = None
        self._lags = self._compute_lags()
        self._phi_ijl = None
        self._norm_ijl = None
        self._ijl2index = None
        self._index2ijl = None
        self._n_index = 0
        self._mark_probabilities = None
        self._mark_probabilities_N = None
        self._mark_min = None
        self._mark_max = None
        self._computed = False
        self._quad_x = None
        self._quad_w = None
        self.set_model(model if model is not None else {})

    def set_model(self, symmetries1d=None, symmetries2d=None, delayed_component=_UNSET, **kwargs):
        if isinstance(symmetries1d, dict) and symmetries2d is None and delayed_component is self._UNSET:
            model = dict(symmetries1d)
            symmetries1d = model.pop("symmetries1d", None)
            symmetries2d = model.pop("symmetries2d", None)
            delayed_component = model.pop("delayed_component", self._UNSET)
            kwargs.update(model)
        if "delayed_component" in kwargs and delayed_component is self._UNSET:
            delayed_component = kwargs.pop("delayed_component")
        kwargs.pop("n_nodes", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown model parameter(s): {unknown}")
        self.symmetries1d = [] if symmetries1d is None else list(symmetries1d)
        self.symmetries2d = [] if symmetries2d is None else list(symmetries2d)
        if delayed_component is not self._UNSET:
            self.delayed_component = (
                None if delayed_component is None else np.asarray(np.atleast_1d(delayed_component), dtype=int)
            )
            self._computed = False
        return self

    def fit(self, events, T=None, end_times=None):
        if end_times is not None:
            if T is not None:
                raise ValueError("provide only one of T or end_times")
            T = end_times
        data, marks, end_times_arr, n_nodes = self._normalize_marked_events(events, T)
        self.data = []
        self._marks = []
        self._end_times = np.empty(0, dtype=float)
        self._n_nodes = int(n_nodes)
        self._fitted = True
        self._computed = False
        self.mark_functions = None
        self._init_marked_components()
        self._init_index()
        for realization, realization_marks, end_time in zip(data, marks, end_times_arr):
            self._append_realization(realization, realization_marks, float(end_time))
        return self.compute()

    def incremental_fit(self, realization, T=None, compute=True, end_times=None):
        if end_times is not None:
            if T is not None:
                raise ValueError("provide only one of T or end_times")
            T = end_times
        new_data, new_marks, new_T, n_nodes = self._normalize_marked_events(realization, T)
        if self.data is None:
            self.data = []
            self._marks = []
            self._end_times = np.empty(0, dtype=float)
            self._n_nodes = int(n_nodes)
            self._fitted = True
            self._init_marked_components()
            self._init_index()
        elif int(n_nodes) != self._n_nodes:
            raise ValueError(f"Bad dimension for realization, should be {self._n_nodes} instead of {n_nodes}")

        if compute and self._has_been_computed_once():
            warnings.warn(
                "compute() method was already called, computed kernels will be updated.",
                UserWarning,
                stacklevel=2,
            )

        for new_realization, realization_marks, end_time in zip(new_data, new_marks, new_T):
            self._append_realization(new_realization, realization_marks, float(end_time))
        if compute:
            return self.compute()
        return self

    def compute(self):
        if self.data is None or self._end_times is None or len(self.data) == 0:
            raise ValueError("no realizations have been added")
        self._estimate_empirical_kernels()
        self._computed = True
        self._fitted = True
        return self

    def _normalize_marked_events(self, events, end_times):
        if events is None or len(events) == 0:
            raise ValueError("events must contain at least one realization")

        if self._is_single_realization(events):
            raw_realizations = [events]
        else:
            raw_realizations = list(events)

        data = []
        marks = []
        n_nodes = None
        for r, realization in enumerate(raw_realizations):
            if realization is None or len(realization) == 0:
                raise ValueError(f"realization {r} must contain at least one node")
            clean_realization = []
            clean_marks = []
            for node, node_value in enumerate(realization):
                timestamps, cumulative_marks = self._coerce_node_observation(node_value, r, node)
                clean_realization.append(timestamps)
                clean_marks.append(cumulative_marks)
            if n_nodes is None:
                n_nodes = len(clean_realization)
            elif len(clean_realization) != n_nodes:
                raise ValueError("all realizations must have the same number of nodes")
            clean_realization, clean_marks = self._delay_realization(clean_realization, clean_marks)
            data.append(clean_realization)
            marks.append(clean_marks)

        clean_end_times = self._normalize_conditional_end_times(data, end_times)
        return data, marks, clean_end_times, int(n_nodes)

    @classmethod
    def _is_single_realization(cls, events):
        first = events[0]
        return cls._is_unmarked_node_observation(first) or cls._is_marked_node_observation(first)

    @staticmethod
    def _is_unmarked_node_observation(value):
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (list, tuple)):
            return len(value) == 0 or isinstance(value[0], (int, float, np.floating))
        return False

    @staticmethod
    def _is_marked_node_observation(value):
        return isinstance(value, tuple) and len(value) == 2

    @classmethod
    def _coerce_node_observation(cls, value, realization_index, node_index):
        if cls._is_marked_node_observation(value):
            timestamps = np.asarray(value[0], dtype=float)
            cumulative_marks = np.asarray(value[1], dtype=float)
        else:
            timestamps = np.asarray(value, dtype=float)
            cumulative_marks = np.arange(1, timestamps.size + 1, dtype=float)
        if timestamps.ndim != 1:
            raise ValueError(f"events[{realization_index}][{node_index}] timestamps must be one-dimensional")
        if cumulative_marks.ndim != 1:
            raise ValueError(f"events[{realization_index}][{node_index}] marks must be one-dimensional")
        if cumulative_marks.size != timestamps.size:
            raise ValueError(f"events[{realization_index}][{node_index}] marks must match timestamps length")
        if timestamps.size and np.any(np.diff(timestamps) < 0):
            raise ValueError(f"timestamps for realization {realization_index}, node {node_index} must be sorted")
        if timestamps.size and timestamps[0] < 0:
            raise ValueError("timestamps must be non-negative")
        if cumulative_marks.size and not np.all(np.isfinite(cumulative_marks)):
            raise ValueError("marks must be finite")
        return timestamps.astype(float, copy=True), cumulative_marks.astype(float, copy=True)

    def _delay_realization(self, realization, marks):
        if self.delayed_component is None:
            return realization, marks
        delayed_realization = [node_events.copy() for node_events in realization]
        delayed_marks = [node_marks.copy() for node_marks in marks]
        for component in self.delayed_component:
            component = int(component)
            if component < 0 or component >= len(delayed_realization):
                raise ValueError("delayed_component contains an invalid node index")
            delayed_realization[component] = delayed_realization[component] + self.delay
            order = np.argsort(delayed_realization[component], kind="mergesort")
            delayed_realization[component] = delayed_realization[component][order]
            delayed_marks[component] = delayed_marks[component][order]
        return delayed_realization, delayed_marks

    @staticmethod
    def _normalize_conditional_end_times(realizations, end_times):
        if end_times is None:
            return np.asarray(
                [
                    max((float(timestamps[-1]) for timestamps in realization if timestamps.size), default=0.0)
                    for realization in realizations
                ],
                dtype=float,
            )
        if isinstance(end_times, (int, float, np.floating)):
            arr = np.asarray([float(end_times)], dtype=float)
        else:
            arr = np.asarray(list(end_times), dtype=float)
        if arr.ndim != 1:
            raise ValueError("end_times must be a scalar or one-dimensional array")
        if arr.size == 1 and len(realizations) > 1:
            arr = np.repeat(arr, len(realizations))
        if arr.size != len(realizations):
            raise ValueError(f"end_times must have length {len(realizations)}, got {arr.size}")
        if np.any(arr < 0):
            raise ValueError("end_times must be non-negative")
        for r, realization in enumerate(realizations):
            latest = max((float(ts[-1]) for ts in realization if ts.size), default=0.0)
            if arr[r] < latest:
                raise ValueError(
                    f"Argument T ({arr[r]:g}) specified is too small, "
                    f"you should use default value or a value greater or equal to {latest:g}."
                )
        return arr

    def _append_realization(self, realization, marks, end_time):
        latest = max((float(node_events[-1]) for node_events in realization if node_events.size), default=-1.0)
        if latest < 0:
            warnings.warn(
                "An empty realization was passed. No computation was performed.",
                UserWarning,
                stacklevel=2,
            )
            return
        if end_time < latest:
            raise ValueError(
                f"Argument T ({end_time:g}) specified is too small, "
                f"you should use default value or a value greater or equal to {latest:g}."
            )
        stored = [np.asarray(node_events, dtype=float).copy() for node_events in realization]
        stored_marks = [np.asarray(node_marks, dtype=float).copy() for node_marks in marks]
        self.data.append(stored)
        self._marks.append(stored_marks)
        self._end_times = np.append(self._end_times, float(end_time))

    def _compute_lags(self):
        if self.claw_method == "log":
            y1 = np.arange(0.0, self.min_lag, self.min_lag * self.delta_lag)
            y2 = np.exp(np.arange(np.log(self.min_lag), np.log(self.max_lag), self.delta_lag))
            lags = np.append(y1, y2)
        else:
            lags = np.arange(0.0, self.max_lag, self.delta_lag)
        if lags.size < 2:
            lags = np.array([0.0, self.max_lag], dtype=float)
        return lags.astype(float)

    def _init_marked_components(self):
        self.marked_components = []
        for node in range(self.n_nodes):
            if node in self._marked_components_spec:
                cuts = np.asarray(self._marked_components_spec[node], dtype=float)
                if cuts.ndim != 1 or cuts.size == 0:
                    raise ValueError("marked component boundaries must be a non-empty one-dimensional array")
                if np.any(np.diff(cuts) <= 0):
                    raise ValueError("marked component boundaries must be strictly increasing")
                intervals = [[-np.inf, float(cuts[0])]]
                intervals.extend([[float(cuts[k]), float(cuts[k + 1])] for k in range(cuts.size - 1)])
                intervals.append([float(cuts[-1]), np.inf])
            else:
                intervals = [[-np.inf, np.inf]]
            self.marked_components.append(intervals)

    def _init_index(self):
        self._ijl2index = []
        self._index2ijl = []
        index = 0
        for i in range(self.n_nodes):
            self._ijl2index.append([])
            for j in range(self.n_nodes):
                self._ijl2index[i].append([])
                for l in range(len(self.marked_components[j])):
                    self._ijl2index[i][j].append(index)
                    self._index2ijl.append((i, j, l))
                    index += 1
        self._n_index = index

    def _quadrature_grid(self):
        method = str(self.quad_method).lower()
        if method in {"gauss", "gauss-"}:
            points, weights = leggauss(self.n_quad)
            x = self.max_support * (points + 1.0) / 2.0
            w = weights * self.max_support / 2.0
        elif method == "lin":
            x = np.arange(0.0, self.max_support, self.max_support / self.n_quad)
            if x.size == 0:
                x = np.array([0.0], dtype=float)
            w = np.empty_like(x)
            w[:-1] = np.diff(x)
            w[-1] = self.max_support / self.n_quad if x.size == 1 else w[-2]
        elif method == "log":
            logstep = (np.log(self.max_support) - np.log(self.min_support) + 1.0) / self.n_quad
            x1 = np.arange(0.0, self.min_support, self.min_support * logstep)
            x2 = np.exp(np.arange(np.log(self.min_support), np.log(self.max_support), logstep))
            x = np.append(x1, x2)
            x = np.unique(np.clip(x, 0.0, self.max_support))
            if x.size == 0:
                x = np.array([0.0], dtype=float)
            w = np.empty_like(x)
            w[:-1] = np.diff(x)
            w[-1] = w[-2] if x.size > 1 else self.max_support
            self.n_quad = int(x.size)
        else:
            raise ValueError("quad_method must be one of 'gauss', 'gauss-', 'lin', or 'log'")

        edges = self._edges_from_quadrature_points(x)
        return x.astype(float), w.astype(float), edges.astype(float)

    def _edges_from_quadrature_points(self, x):
        if x.size == 1:
            return np.array([0.0, self.max_support], dtype=float)
        edges = np.empty(x.size + 1, dtype=float)
        edges[0] = 0.0
        edges[-1] = self.max_support
        edges[1:-1] = (x[:-1] + x[1:]) / 2.0
        edges = np.maximum.accumulate(edges)
        if edges[-1] <= edges[0]:
            edges[-1] = self.max_support
        return edges

    def _estimate_empirical_kernels(self):
        if len(self.data) == 0:
            raise ValueError("no non-empty realizations have been added")
        total_time = float(np.sum(self._end_times))
        counts = np.zeros(self.n_nodes, dtype=float)
        for realization, end_time in zip(self.data, self._end_times):
            for u in range(self.n_nodes):
                timestamps = realization[u]
                counts[u] += np.count_nonzero((timestamps >= 0.0) & (timestamps <= end_time))
        self.mean_intensity = counts / max(total_time, 1e-15)
        self._apply_symmetries_1d(self.mean_intensity)

        self._quad_x, self._quad_w, edges = self._quadrature_grid()
        n_quad = self._quad_x.size
        n_marks = [len(intervals) for intervals in self.marked_components]
        bin_counts = [
            [np.zeros((n_marks[j], n_quad), dtype=float) for j in range(self.n_nodes)]
            for _ in range(self.n_nodes)
        ]
        exposures = [np.zeros((n_marks[j], n_quad), dtype=float) for j in range(self.n_nodes)]
        mark_counts = [np.zeros(n_marks[j], dtype=float) for j in range(self.n_nodes)]
        self._mark_min = np.full(self.n_nodes, np.inf, dtype=float)
        self._mark_max = np.full(self.n_nodes, -np.inf, dtype=float)

        for realization, realization_marks, end_time in zip(self.data, self._marks, self._end_times):
            end_time = float(end_time)
            valid_events = []
            valid_mark_increments = []
            for node_events, node_marks in zip(realization, realization_marks):
                mask = (node_events >= 0.0) & (node_events <= end_time)
                valid_events.append(np.asarray(node_events[mask], dtype=float))
                valid_cumulative_marks = np.asarray(node_marks[mask], dtype=float)
                if valid_cumulative_marks.size == 0:
                    valid_mark_increments.append(np.asarray([], dtype=float))
                else:
                    valid_mark_increments.append(
                        np.hstack((valid_cumulative_marks[0], np.diff(valid_cumulative_marks)))
                    )

            for j in range(self.n_nodes):
                source_times = valid_events[j]
                source_marks = valid_mark_increments[j]
                if source_marks.size:
                    self._mark_min[j] = min(self._mark_min[j], float(np.min(source_marks)))
                    self._mark_max[j] = max(self._mark_max[j], float(np.max(source_marks)))
                for l, interval in enumerate(self.marked_components[j]):
                    low, high = interval
                    mark_mask = (source_marks >= low) & (source_marks < high)
                    marked_source_times = source_times[mark_mask]
                    mark_counts[j][l] += marked_source_times.size
                    source_times_before_end = marked_source_times[marked_source_times < end_time]
                    for tj in source_times_before_end:
                        remaining = end_time - float(tj)
                        right = np.minimum(edges[1:], remaining)
                        exposures[j][l] += np.maximum(0.0, right - edges[:-1])

                    for i in range(self.n_nodes):
                        target_times = valid_events[i]
                        for tj in source_times_before_end:
                            start = np.searchsorted(target_times, tj, side="right")
                            stop = np.searchsorted(
                                target_times,
                                min(float(tj) + self.max_support, end_time),
                                side="right",
                            )
                            if stop <= start:
                                continue
                            lags = target_times[start:stop] - float(tj)
                            if lags.size:
                                bin_counts[i][j][l] += np.histogram(lags, bins=edges)[0]

        self._mark_probabilities_N = [counts_j.copy() for counts_j in mark_counts]
        self._mark_probabilities = []
        for j, counts_j in enumerate(mark_counts):
            total = float(np.sum(counts_j))
            if total > 0:
                self._mark_probabilities.append(counts_j / total)
            else:
                probs = np.zeros_like(counts_j)
                if probs.size:
                    probs[0] = 1.0
                self._mark_probabilities.append(probs)
            if not np.isfinite(self._mark_min[j]):
                self._mark_min[j] = 1.0
                self._mark_max[j] = 1.0

        self._apply_mark_symmetries()

        values_by_mark = [[[] for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]
        aggregate_values = np.zeros((self.n_nodes, self.n_nodes, n_quad), dtype=float)
        aggregate_norms = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        use_trapezoid_norm = str(self.quad_method).lower() in {"lin", "log"}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                for l in range(n_marks[j]):
                    empirical_rate = np.divide(
                        bin_counts[i][j][l],
                        exposures[j][l],
                        out=np.zeros(n_quad, dtype=float),
                        where=exposures[j][l] > 0,
                    )
                    y_l = empirical_rate - self.mean_intensity[i]
                    values_by_mark[i][j].append(y_l)
                    prob = self._mark_probabilities[j][l]
                    aggregate_values[i, j] += prob * y_l
                    aggregate_norms[i, j] += prob * self._kernel_norm(y_l, use_trapezoid_norm)

        self._apply_symmetries_2d_to_marked_values(values_by_mark)
        aggregate_values = np.zeros((self.n_nodes, self.n_nodes, n_quad), dtype=float)
        aggregate_norms = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        self._phi_ijl = []
        self._norm_ijl = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                for l, y_l in enumerate(values_by_mark[i][j]):
                    norm_l = self._kernel_norm(y_l, use_trapezoid_norm)
                    self._phi_ijl.append((self._quad_x.copy(), y_l.copy()))
                    self._norm_ijl.append(norm_l)
                    prob = self._mark_probabilities[j][l]
                    aggregate_values[i, j] += prob * y_l
                    aggregate_norms[i, j] += prob * norm_l

        self._apply_symmetries_2d(aggregate_values)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                aggregate_norms[i, j] = self._kernel_norm(aggregate_values[i, j], use_trapezoid_norm)

        self.kernels = []
        self.kernels_norms = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            self.kernels.append([])
            for j in range(self.n_nodes):
                y = aggregate_values[i, j].copy()
                self.kernels[i].append(np.vstack((self._quad_x.copy(), y)))
                self.kernels_norms[i, j] = aggregate_norms[i, j]
        self.baseline = (np.eye(self.n_nodes) - self.kernels_norms).dot(self.mean_intensity)
        self._estimate_mark_functions()

    def _kernel_norm(self, values, use_trapezoid_norm):
        if use_trapezoid_norm and values.size > 1:
            return float(np.sum((values[:-1] + values[1:]) * self._quad_w[:-1] / 2.0))
        return float(np.sum(values * self._quad_w))

    def _apply_mark_symmetries(self):
        for group in self.symmetries1d:
            indices = self._as_index_group(group)
            if not indices:
                continue
            if len({len(self.marked_components[index]) for index in indices}) != 1:
                continue
            avg_counts = np.mean([self._mark_probabilities_N[index] for index in indices], axis=0)
            total = float(np.sum(avg_counts))
            avg_probs = avg_counts / total if total > 0 else np.zeros_like(avg_counts)
            avg_min = float(np.mean([self._mark_min[index] for index in indices]))
            avg_max = float(np.mean([self._mark_max[index] for index in indices]))
            for index in indices:
                self._mark_probabilities_N[index] = avg_counts.copy()
                self._mark_probabilities[index] = avg_probs.copy()
                self._mark_min[index] = avg_min
                self._mark_max[index] = avg_max

    def _apply_symmetries_2d_to_marked_values(self, values_by_mark):
        for group in self.symmetries2d:
            pairs = self._as_pair_group(group)
            if not pairs:
                continue
            if len({len(values_by_mark[i][j]) for i, j in pairs}) != 1:
                continue
            for l in range(len(values_by_mark[pairs[0][0]][pairs[0][1]])):
                avg = np.mean([values_by_mark[i][j][l] for i, j in pairs], axis=0)
                for i, j in pairs:
                    values_by_mark[i][j][l] = avg.copy()

    def _estimate_mark_functions(self):
        self.mark_functions = []
        for i in range(self.n_nodes):
            self.mark_functions.append([])
            for j in range(self.n_nodes):
                intervals = self.marked_components[j]
                if len(intervals) == 1:
                    self.mark_functions[i].append((np.array([1.0]), np.array([1.0])))
                    continue
                x_parts = []
                y_parts = []
                n_points = 100
                denominator = self.kernels_norms[i, j]
                for l, interval in enumerate(intervals):
                    index = self._ijl2index[i][j][l]
                    ratio = 0.0 if abs(denominator) <= 1e-15 else self._norm_ijl[index] / denominator
                    xmin, xmax = interval
                    if l == 0:
                        xmin = self._mark_min[j]
                    if l == len(intervals) - 1:
                        xmax = self._mark_max[j]
                    if not np.isfinite(xmin):
                        xmin = self._mark_min[j]
                    if not np.isfinite(xmax):
                        xmax = self._mark_max[j]
                    if xmax < xmin:
                        xmin, xmax = xmax, xmin
                    if math.isclose(float(xmin), float(xmax)):
                        x = np.full(n_points, float(xmin), dtype=float)
                    else:
                        x = np.linspace(float(xmin), float(xmax), n_points)
                    x_parts.append(x)
                    y_parts.append(np.full(n_points, float(ratio), dtype=float))
                self.mark_functions[i].append((np.concatenate(x_parts), np.concatenate(y_parts)))

    def _apply_symmetries_1d(self, vector):
        for group in self.symmetries1d:
            indices = self._as_index_group(group)
            if not indices:
                continue
            avg = float(np.mean(vector[indices]))
            vector[indices] = avg

    def _apply_symmetries_2d(self, values):
        for group in self.symmetries2d:
            pairs = self._as_pair_group(group)
            if not pairs:
                continue
            avg = np.mean([values[i, j] for i, j in pairs], axis=0)
            for i, j in pairs:
                values[i, j] = avg

    @staticmethod
    def _as_index_group(group):
        if np.isscalar(group):
            return [int(group)]
        return [int(index) for index in group]

    @staticmethod
    def _as_pair_group(group):
        if len(group) == 2 and np.isscalar(group[0]) and np.isscalar(group[1]):
            return [(int(group[0]), int(group[1]))]
        return [(int(pair[0]), int(pair[1])) for pair in group]

    def get_kernel_supports(self):
        if self.kernels is None:
            raise ValueError("compute must be called first")
        supports = np.empty((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                supports[i, j] = float(np.max(self.kernels[i][j][0]))
        return supports

    def get_kernel_values(self, i, j, abscissa_array):
        if self.kernels is None:
            raise ValueError("compute must be called first")
        x = np.asarray(abscissa_array, dtype=float)
        t_values = self.kernels[i][j][0]
        y_values = self.kernels[i][j][1]
        if str(self.quad_method).lower() == "log":
            out = np.zeros_like(x)
            mask = x > 0
            positive_t = t_values > 0
            positive_y = y_values > 0
            if np.any(positive_t) and np.any(positive_y):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[mask] = np.power(
                        10.0,
                        np.interp(
                            np.log10(x[mask]),
                            np.log10(t_values[positive_t]),
                            np.log10(np.maximum(y_values[positive_t], 1e-300)),
                            left=-300.0,
                            right=-300.0,
                        ),
                    )
            return out
        return np.interp(x, t_values, y_values, left=0.0, right=0.0)

    def get_kernel_norms(self):
        if self.kernels_norms is None:
            raise ValueError("compute must be called first")
        return self.kernels_norms.copy()

    def _has_been_computed_once(self):
        return bool(self._computed)


class HawkesCumulantMatching(_LearnerBase):
    """Cumulant-matching estimator for kernel norms."""

    def __init__(
        self,
        integration_support,
        C: float = 1e3,
        penalty: str = "none",
        solver: str = "adam",
        step: float = 1e-2,
        tol: float = 1e-8,
        max_iter: int = 1000,
        verbose: bool = False,
        print_every: int = 100,
        record_every: int = 10,
        solver_kwargs=None,
        cs_ratio=None,
        elastic_net_ratio: float = 0.95,
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every)
        self.integration_support = integration_support
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.step = step
        self.solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        self.cs_ratio = cs_ratio
        self.elastic_net_ratio = elastic_net_ratio
        self.solution = None
        self._mean_intensity = None
        self._covariance = None
        self._skewness = None

    def fit(self, events, end_times=None, adjacency_start=None, R_start=None):
        self._set_data(events, end_times)
        self.compute_cumulants(force=True)
        if adjacency_start is not None and not isinstance(adjacency_start, str):
            adjacency = np.asarray(adjacency_start, dtype=float)
            self.solution = np.linalg.inv(np.eye(self.n_nodes) - adjacency)
        elif R_start is not None:
            self.solution = np.asarray(R_start, dtype=float)
        else:
            self.solution = np.eye(self.n_nodes)
        return self

    def compute_cumulants(self, force=False):
        if self._mean_intensity is not None and not force:
            return
        total_time = float(np.sum(self._end_times))
        counts = np.asarray([sum(r[u].size for r in self.data) for u in range(self.n_nodes)], dtype=float)
        self._mean_intensity = counts / max(total_time, 1e-15)
        centered_counts = []
        for realization, end_time in zip(self.data, self._end_times):
            del end_time
            centered_counts.append([realization[u].size for u in range(self.n_nodes)])
        centered_counts = np.asarray(centered_counts, dtype=float)
        if centered_counts.shape[0] > 1:
            self._covariance = np.cov(centered_counts, rowvar=False)
        else:
            self._covariance = np.diag(np.maximum(self._mean_intensity, 1e-12))
        self._skewness = np.zeros((self.n_nodes, self.n_nodes), dtype=float)

    @property
    def mean_intensity(self):
        self.compute_cumulants()
        return self._mean_intensity.copy()

    @property
    def covariance(self):
        self.compute_cumulants()
        return self._covariance.copy()

    @property
    def skewness(self):
        self.compute_cumulants()
        return self._skewness.copy()

    @property
    def adjacency(self):
        return np.eye(self.n_nodes) - scipy.linalg.inv(self.solution)

    @property
    def baseline(self):
        return scipy.linalg.inv(self.solution).dot(self.mean_intensity)

    def objective(self, adjacency=None, R=None):
        if adjacency is not None:
            R = np.linalg.inv(np.eye(self.n_nodes) - np.asarray(adjacency, dtype=float))
        elif R is None:
            R = self.solution
        target = self.covariance
        predicted = R @ np.diag(self.mean_intensity) @ R.T
        return float(np.linalg.norm(predicted - target) ** 2)

    def approximate_optimal_cs_ratio(self):
        norm_c = np.linalg.norm(self.covariance) ** 2
        norm_k = np.linalg.norm(self.skewness) ** 2
        return float(norm_k / max(norm_k + norm_c, 1e-15))

    def starting_point(self, random=False):
        sqrt_c = scipy.linalg.sqrtm(self.covariance)
        sqrt_l = np.sqrt(np.maximum(self.mean_intensity, 1e-12))
        if random:
            q, _ = scipy.linalg.qr(np.random.rand(self.n_nodes, self.n_nodes))
        else:
            q = np.eye(self.n_nodes)
        return np.real_if_close(sqrt_c @ q @ np.diag(1.0 / sqrt_l))

    def get_kernel_values(self, i, j, abscissa_array):
        raise ValueError("Hawkes cumulant matching estimates kernel norms only")

    def get_kernel_norms(self):
        return self.adjacency


class HawkesCumulantMatchingPyT(HawkesCumulantMatching):
    """PyTorch-backed compatibility class.

    The current implementation uses the shared NumPy estimator unless a future
    PyTorch optimizer is requested explicitly through solver_kwargs.
    """

    def __init__(self, *args, **kwargs):
        import torch  # noqa: F401

        super().__init__(*args, **kwargs)


class HawkesCumulantMatchingTf(HawkesCumulantMatching):
    """TensorFlow optional backend."""

    def __init__(self, *args, **kwargs):
        try:
            import tensorflow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HawkesCumulantMatchingTf requires TensorFlow, which is not installed"
            ) from exc
        super().__init__(*args, **kwargs)


def _piecewise_loglik(data, end_times, baseline, kernel, discretization):
    value = 0.0
    n_jumps = 0
    for realization, end_time in zip(data, end_times):
        value += baseline.size * float(end_time)
        for u in range(baseline.size):
            value -= _piecewise_compensator_at(float(end_time), u, realization, baseline, kernel, discretization)
            for t in realization[u]:
                n_jumps += 1
                intensity = baseline[u]
                for v in range(baseline.size):
                    for tj in realization[v]:
                        if tj >= t:
                            break
                        lag = float(t - tj)
                        if lag >= discretization[-1]:
                            continue
                        m = int(np.searchsorted(discretization, lag) - 1)
                        if 0 <= m < kernel.shape[2]:
                            intensity += kernel[u, v, m]
                if intensity > 0:
                    value += np.log(intensity)
                else:
                    return -np.inf
    return float(value / max(n_jumps, 1))


def _piecewise_compensator_at(t, u, realization, baseline, kernel, discretization):
    value = float(baseline[u] * t)
    for v in range(baseline.size):
        for tj in realization[v]:
            if tj >= t:
                break
            remaining = t - float(tj)
            for m in range(kernel.shape[2]):
                left = discretization[m]
                right = discretization[m + 1]
                overlap = max(0.0, min(remaining, right) - left)
                if overlap > 0:
                    value += kernel[u, v, m] * overlap
    return value
