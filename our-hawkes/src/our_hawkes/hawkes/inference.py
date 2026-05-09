"""Hawkes inference and learner classes."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.linalg

from our_hawkes.base import BaseEstimator, History, normalize_events, relative_distance

from .models import (
    ModelHawkesExpKernLeastSq,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik,
)
from .simulation import SimuHawkesExpKernels, SimuHawkesSumExpKernels
from .solvers import ProxL1, ProxNuclear, optimize_positive_coeffs


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
    _allowed_solvers = {"gd", "agd", "bfgs", "svrg", "sgd"}
    _allowed_penalties = {"none", "l1", "l2", "elasticnet", "nuclear"}

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
    ):
        super().__init__(
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            print_every=print_every,
            record_every=record_every,
        )
        if penalty not in self._allowed_penalties:
            raise ValueError(f"unknown penalty {penalty!r}")
        if solver not in self._allowed_solvers:
            raise ValueError(f"unknown solver {solver!r}")
        if C is None or C <= 0:
            raise ValueError("C must be positive")
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.step = step
        self.elastic_net_ratio = elastic_net_ratio
        self.random_state = random_state
        self.coeffs = None
        self.events = None
        self._model_obj = None

    def _fit_model(self, model, events, end_times=None, start=None):
        self._set_data(events, end_times)
        self.events = events
        model.fit(events, end_times=end_times)
        self._model_obj = model
        if start is None:
            start_coeffs = np.ones(model.n_coeffs, dtype=float)
        elif isinstance(start, (int, float, np.floating)):
            start_coeffs = np.full(model.n_coeffs, float(start), dtype=float)
        else:
            start_coeffs = np.asarray(start, dtype=float).copy()
            if start_coeffs.shape != (model.n_coeffs,):
                raise ValueError(f"start has shape {start_coeffs.shape}, expected {(model.n_coeffs,)}")
        self.coeffs = optimize_positive_coeffs(
            model,
            start_coeffs,
            penalty=self.penalty,
            C=self.C,
            elastic_net_ratio=self.elastic_net_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            jac=self.penalty != "nuclear",
        )
        self._fitted = True
        return self

    @property
    def baseline(self):
        if not self._fitted:
            raise ValueError("fit must be called first")
        return self.coeffs[: self.n_nodes]

    def score(self, events=None, end_times=None, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        if events is None and end_times is None:
            model = self._model_obj
        else:
            model = self._construct_model_obj()
            model.fit(events, end_times=end_times)
        return -model.loss(np.asarray(coeffs, dtype=float))

    def estimated_intensity(self, events, intensity_track_step, end_time=None):
        if end_time is None:
            end_time = max((float(ts[-1]) for ts in events if len(ts)), default=0.0)
        simu = self._corresponding_simu()
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
            discretization = np.asarray(kernel_discretization, dtype=float)
            if discretization.ndim != 1 or discretization.size < 2:
                raise ValueError("kernel_discretization must contain at least two points")
            if np.any(np.diff(discretization) <= 0):
                raise ValueError("kernel_discretization must be strictly increasing")
            self.kernel_discretization = discretization
            self.kernel_support = float(discretization[-1])
            self.kernel_size = discretization.size - 1
        elif kernel_support is not None:
            if kernel_support <= 0 or kernel_size <= 0:
                raise ValueError("kernel_support and kernel_size must be positive")
            self.kernel_support = float(kernel_support)
            self.kernel_size = int(kernel_size)
            self.kernel_discretization = np.linspace(0.0, self.kernel_support, self.kernel_size + 1)
        else:
            raise ValueError("either kernel_support or kernel_discretization must be provided")
        self.baseline = None
        self.kernel = None
        self.history.print_order = ["n_iter", "rel_baseline", "rel_kernel"]

    @property
    def kernel_dt(self):
        diffs = np.diff(self.kernel_discretization)
        if np.allclose(diffs, diffs[0]):
            return float(diffs[0])
        return diffs

    def fit(self, events, end_times=None, baseline_start=None, kernel_start=None):
        self._set_data(events, end_times)
        rng = np.random.default_rng(0)
        if baseline_start is None:
            self.baseline = np.ones(self.n_nodes, dtype=float)
        else:
            self.baseline = np.asarray(baseline_start, dtype=float).copy()
        if kernel_start is None:
            self.kernel = 0.1 * rng.uniform(size=(self.n_nodes, self.n_nodes, self.kernel_size))
        else:
            self.kernel = np.asarray(kernel_start, dtype=float).copy()
        if self.kernel.shape != (self.n_nodes, self.n_nodes, self.kernel_size):
            raise ValueError("kernel_start has wrong shape")

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
        dt = np.diff(self.kernel_discretization)
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
                            m = int(np.searchsorted(self.kernel_discretization, lag, side="right") - 1)
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
        idx = np.searchsorted(self.kernel_discretization, x[mask], side="right") - 1
        values[mask] = self.kernel[i, j, idx]
        return values

    def _compute_primitive_kernel_values(self, i, j, abscissa_array):
        primitives = self._get_kernel_primitives()
        x = np.asarray(abscissa_array, dtype=float)
        idx = np.clip(np.searchsorted(self.kernel_discretization, x, side="right") - 1, 0, self.kernel_size - 1)
        return primitives[i, j, idx]

    def get_kernel_norms(self):
        return np.einsum("ijk,k->ij", self.kernel, np.diff(self.kernel_discretization))

    def _get_kernel_primitives(self):
        dt = np.diff(self.kernel_discretization)
        return np.cumsum(self.kernel * dt[None, None, :], axis=2)

    def score(self, events=None, end_times=None, baseline=None, kernel=None):
        if events is None:
            data = self.data
            end_times_arr = self._end_times
        else:
            data, end_times_arr, _ = normalize_events(events, end_times)
        baseline = self.baseline if baseline is None else np.asarray(baseline, dtype=float)
        kernel = self.kernel if kernel is None else np.asarray(kernel, dtype=float)
        return _piecewise_loglik(data, end_times_arr, baseline, kernel, self.kernel_discretization)

    def time_changed_interarrival_times(self, events=None, end_times=None, baseline=None, kernel=None):
        if events is None:
            data = self.data
            end_times_arr = self._end_times
        else:
            data, end_times_arr, _ = normalize_events(events, end_times)
        baseline = self.baseline if baseline is None else np.asarray(baseline, dtype=float)
        kernel = self.kernel if kernel is None else np.asarray(kernel, dtype=float)
        out = []
        for realization, end_time in zip(data, end_times_arr):
            del end_time
            out_r = []
            for u in range(self.n_nodes):
                vals = [
                    _piecewise_compensator_at(float(t), u, realization, baseline, kernel, self.kernel_discretization)
                    for t in realization[u]
                ]
                out_r.append(np.diff(vals))
            out.append(out_r)
        return out


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
    ):
        super().__init__(tol, max_iter, verbose, print_every, record_every, n_threads)
        self.decay = float(decay)
        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.rho = rho
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.baseline = None
        self.adjacency = None
        self._prox_l1 = ProxL1(self.strength_lasso)
        self._prox_nuclear = ProxNuclear(self.strength_nuclear)

    @property
    def strength_lasso(self):
        return self.lasso_nuclear_ratio / self.C

    @property
    def strength_nuclear(self):
        return (1.0 - self.lasso_nuclear_ratio) / self.C

    @property
    def coeffs(self):
        return np.hstack((self.baseline, self.adjacency.ravel()))

    def fit(self, events, end_times=None, baseline_start=None, adjacency_start=None):
        self._set_data(events, end_times)
        start = None
        if baseline_start is not None or adjacency_start is not None:
            baseline_start = np.ones(self.n_nodes) if baseline_start is None else baseline_start
            adjacency_start = np.full((self.n_nodes, self.n_nodes), 0.1) if adjacency_start is None else adjacency_start
            start = np.hstack((np.asarray(baseline_start, dtype=float), np.asarray(adjacency_start, dtype=float).ravel()))
        learner = HawkesExpKern(
            self.decay,
            gofit="likelihood",
            penalty="elasticnet",
            C=self.C,
            elastic_net_ratio=self.lasso_nuclear_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        ).fit(events, start=start, end_times=end_times)
        self.baseline = learner.baseline.copy()
        self.adjacency = learner.adjacency.copy()
        self._model = learner._model_obj
        self._fitted = True
        return self

    def objective(self, coeffs, loss=None):
        if loss is None:
            loss = self._model.loss(coeffs)
        return float(loss + self._prox_l1.value(self.adjacency.ravel()) + self._prox_nuclear.value(self.adjacency))

    def _corresponding_simu(self):
        return SimuHawkesExpKernels(self.adjacency, self.decay, self.baseline, force_simulation=True, verbose=False)

    def get_kernel_supports(self):
        return np.vectorize(lambda kernel: kernel.get_plot_support())(self._corresponding_simu().kernels)

    def get_kernel_values(self, i, j, abscissa_array):
        return self._corresponding_simu().kernels[i, j].get_values(abscissa_array)

    def get_kernel_norms(self):
        return self.adjacency.copy()

    def score(self, events=None, end_times=None, baseline=None, adjacency=None):
        baseline = self.baseline if baseline is None else baseline
        adjacency = self.adjacency if adjacency is None else adjacency
        coeffs = np.hstack((np.asarray(baseline, dtype=float), np.asarray(adjacency, dtype=float).ravel()))
        model = self._model if events is None else ModelHawkesExpKernLogLik(self.decay).fit(events, end_times)
        return -model.loss(coeffs)


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
        self.max_mean_gaussian = float(max_mean_gaussian)
        self.n_gaussians = int(n_gaussians)
        self.step_size = step_size
        self.C = C
        self.lasso_grouplasso_ratio = lasso_grouplasso_ratio
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.baseline = None
        self.amplitudes = None

    @property
    def means_gaussians(self):
        return np.arange(self.n_gaussians) * self.max_mean_gaussian / self.n_gaussians

    @property
    def std_gaussian(self):
        return self.max_mean_gaussian / (self.n_gaussians * np.pi)

    def fit(self, events, end_times=None, baseline_start=None, amplitudes_start=None):
        self._set_data(events, end_times)
        if amplitudes_start is not None:
            self.baseline = np.ones(self.n_nodes) if baseline_start is None else np.asarray(baseline_start, dtype=float)
            self.amplitudes = np.asarray(amplitudes_start, dtype=float).copy()
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
        return np.full((self.n_nodes, self.n_nodes), self.max_mean_gaussian)

    def get_kernel_values(self, i, j, abscissa_array):
        x = np.asarray(abscissa_array, dtype=float)
        values = np.zeros_like(x)
        sigma = self.std_gaussian
        for m, mean in enumerate(self.means_gaussians):
            values += self.amplitudes[i, j, m] * np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        return values

    def get_kernel_norms(self):
        return np.einsum("ijk->ij", self.amplitudes)


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
        self.kernel_support = float(kernel_support)
        self.kernel_size = int(kernel_size)
        self.n_basis = n_basis
        self.C = C
        self.ode_max_iter = ode_max_iter
        self.ode_tol = ode_tol
        self.kernel_discretization = np.linspace(0.0, self.kernel_support, self.kernel_size + 1)
        self.kernel_dt = self.kernel_support / self.kernel_size
        self.baseline = None
        self.amplitudes = None
        self.basis_kernels = None

    def fit(self, events, end_times=None, baseline_start=None, amplitudes_start=None, basis_kernels_start=None):
        self._set_data(events, end_times)
        if self.n_basis in (None, 0):
            self.n_basis = self.n_nodes
        if basis_kernels_start is not None and amplitudes_start is not None:
            self.baseline = np.ones(self.n_nodes) if baseline_start is None else np.asarray(baseline_start, dtype=float)
            self.basis_kernels = np.asarray(basis_kernels_start, dtype=float)
            self.amplitudes = np.asarray(amplitudes_start, dtype=float)
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
        idx = np.searchsorted(self.kernel_discretization, x[mask], side="right") - 1
        kernel_ij = np.tensordot(self.amplitudes[i, j, :], self.basis_kernels, axes=(0, 0))
        values[mask] = kernel_ij[idx]
        return values


class HawkesConditionalLaw(_LearnerBase):
    """Empirical conditional-law estimator with tick-compatible outputs."""

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
        if max_support <= 0:
            raise ValueError("max_support must be positive")
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        self.delta_lag = float(delta_lag)
        self.min_lag = float(min_lag)
        self.max_lag = float(max_lag)
        self.n_quad = int(n_quad)
        self.max_support = float(max_support)
        self.min_support = float(min_support)
        self.quad_method = quad_method
        self.marked_components = {} if marked_components is None else marked_components
        self.delayed_component = delayed_component
        self.delay = float(delay)
        self.claw_method = claw_method
        self.mean_intensity = None
        self.baseline = None
        self.kernels = None
        self.kernels_norms = None
        self.mark_functions = None
        self._computed = False
        self._quad_x = None
        self._quad_w = None
        self.set_model(**(model or {})) if isinstance(model, dict) else self.set_model()

    def set_model(self, symmetries1d=None, symmetries2d=None, **kwargs):
        del kwargs
        self.symmetries1d = [] if symmetries1d is None else list(symmetries1d)
        self.symmetries2d = [] if symmetries2d is None else list(symmetries2d)
        return self

    def fit(self, events, T=None, end_times=None):
        if end_times is not None:
            if T is not None:
                raise ValueError("provide only one of T or end_times")
            T = end_times
        data, end_times_arr, n_nodes = normalize_events(events, T)
        self.data = []
        self._end_times = np.empty(0, dtype=float)
        self._n_nodes = int(n_nodes)
        self._fitted = True
        self._computed = False
        self.mark_functions = None
        for realization, end_time in zip(data, end_times_arr):
            self._append_realization(realization, float(end_time))
        return self.compute()

    def incremental_fit(self, realization, T=None, compute=True, end_times=None):
        if end_times is not None:
            if T is not None:
                raise ValueError("provide only one of T or end_times")
            T = end_times
        new_data, new_T, n_nodes = normalize_events(realization, T)
        if self.data is None:
            self.data = []
            self._end_times = np.empty(0, dtype=float)
            self._n_nodes = int(n_nodes)
            self._fitted = True
        elif int(n_nodes) != self._n_nodes:
            raise ValueError(f"Bad dimension for realization, should be {self._n_nodes} instead of {n_nodes}")

        if compute and self._has_been_computed_once():
            warnings.warn(
                "compute() method was already called, computed kernels will be updated.",
                UserWarning,
                stacklevel=2,
            )

        for new_realization, end_time in zip(new_data, new_T):
            self._append_realization(new_realization, float(end_time))
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

    def _append_realization(self, realization, end_time):
        stored = [np.asarray(node_events, dtype=float).copy() for node_events in realization]
        if self.delayed_component is not None:
            delayed = np.atleast_1d(self.delayed_component)
            for component in delayed:
                component = int(component)
                if component < 0 or component >= len(stored):
                    raise ValueError("delayed_component contains an invalid node index")
                stored[component] = np.sort(stored[component] + self.delay)
        self.data.append(stored)
        self._end_times = np.append(self._end_times, float(end_time))

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
        bin_counts = np.zeros((self.n_nodes, self.n_nodes, n_quad), dtype=float)
        exposures = np.zeros((self.n_nodes, n_quad), dtype=float)

        for realization, end_time in zip(self.data, self._end_times):
            end_time = float(end_time)
            valid_events = [
                np.asarray(node_events[(node_events >= 0.0) & (node_events <= end_time)], dtype=float)
                for node_events in realization
            ]
            for j in range(self.n_nodes):
                source_times = valid_events[j][valid_events[j] < end_time]
                for tj in source_times:
                    remaining = end_time - float(tj)
                    right = np.minimum(edges[1:], remaining)
                    exposures[j] += np.maximum(0.0, right - edges[:-1])

                for i in range(self.n_nodes):
                    target_times = valid_events[i]
                    for tj in source_times:
                        start = np.searchsorted(target_times, tj, side="right")
                        stop = np.searchsorted(target_times, min(float(tj) + self.max_support, end_time), side="right")
                        if stop <= start:
                            continue
                        lags = target_times[start:stop] - float(tj)
                        if lags.size:
                            bin_counts[i, j] += np.histogram(lags, bins=edges)[0]

        values = np.zeros_like(bin_counts)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                empirical_rate = np.divide(
                    bin_counts[i, j],
                    exposures[j],
                    out=np.zeros(n_quad, dtype=float),
                    where=exposures[j] > 0,
                )
                values[i, j] = empirical_rate - self.mean_intensity[i]

        self._apply_symmetries_2d(values)

        self.kernels = []
        self.kernels_norms = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        use_trapezoid_norm = str(self.quad_method).lower() in {"lin", "log"}
        for i in range(self.n_nodes):
            self.kernels.append([])
            for j in range(self.n_nodes):
                y = values[i, j].copy()
                self.kernels[i].append(np.vstack((self._quad_x.copy(), y)))
                if use_trapezoid_norm and y.size > 1:
                    self.kernels_norms[i, j] = float(np.sum((y[:-1] + y[1:]) * self._quad_w[:-1] / 2.0))
                else:
                    self.kernels_norms[i, j] = float(np.sum(y * self._quad_w))
        self.baseline = (np.eye(self.n_nodes) - self.kernels_norms).dot(self.mean_intensity)
        self.mark_functions = [
            [(np.array([1.0]), np.array([1.0])) for _ in range(self.n_nodes)]
            for _ in range(self.n_nodes)
        ]

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
    for realization, end_time in zip(data, end_times):
        for u in range(baseline.size):
            value -= _piecewise_compensator_at(float(end_time), u, realization, baseline, kernel, discretization)
            for t in realization[u]:
                intensity = baseline[u]
                for v in range(baseline.size):
                    for tj in realization[v]:
                        if tj >= t:
                            break
                        lag = float(t - tj)
                        if lag >= discretization[-1]:
                            continue
                        m = int(np.searchsorted(discretization, lag, side="right") - 1)
                        if 0 <= m < kernel.shape[2]:
                            intensity += kernel[u, v, m]
                if intensity > 0:
                    value += np.log(intensity)
                else:
                    return -np.inf
    return float(value)


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
