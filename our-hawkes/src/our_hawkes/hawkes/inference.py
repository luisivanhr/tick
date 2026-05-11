"""Hawkes inference and learner classes."""

from __future__ import annotations

import math
import warnings
from itertools import product
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
from .numeric import NUMBA_AVAILABLE, _compile, pack_realization, pack_realizations
from .simulation import SimuHawkesExpKernels, SimuHawkesSumExpKernels
from .solvers import (
    AGD,
    BFGS,
    GD,
    ProxElasticNet,
    ProxL1,
    ProxL2Sq,
    ProxNuclear,
    ProxPositive,
    SGD,
    SVRG,
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
        print_every = max(int(self.print_every), 1)
        record_every = max(int(self.record_every), 1)
        if self.max_iter < print_every and self.max_iter < record_every:
            return False
        if i % print_every == 0 or i % record_every == 0:
            return True
        if i + 1 == self.max_iter:
            return True
        return False

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

    def plot_estimated_intensity(self, events=None, intensity_track_step=None, end_time=None, **kwargs):
        from our_hawkes.plot import plot_estimated_intensity

        return plot_estimated_intensity(
            self,
            events=events,
            intensity_track_step=intensity_track_step,
            end_time=end_time,
            **kwargs,
        )

    def qq_plots(self, events=None, end_times=None, **kwargs):
        from our_hawkes.plot import qq_plots

        if hasattr(self, "time_changed_interarrival_times"):
            residuals_by_realization = self.time_changed_interarrival_times(
                events=events,
                end_times=end_times,
            )
            n_nodes = len(residuals_by_realization[0]) if residuals_by_realization else self.n_nodes
            residuals = []
            for node in range(n_nodes):
                pieces = [
                    np.asarray(realization[node], dtype=float)
                    for realization in residuals_by_realization
                    if len(realization[node])
                ]
                residuals.append(np.concatenate(pieces) if pieces else np.asarray([], dtype=float))
            return qq_plots(residuals=residuals, **kwargs)

        if events is None:
            if self.data is None:
                raise ValueError("events must be provided unless the learner has been fitted")
            events = self.data
            end_times = self._end_times if end_times is None else end_times
        data, normalized_end_times, _ = normalize_events(events, end_times)
        node_residuals = [list() for _ in range(self.n_nodes)]
        for realization, realization_end_time in zip(data, normalized_end_times):
            simu = self._corresponding_simu()
            simu.set_timestamps(realization, end_time=float(realization_end_time))
            simu.store_compensator_values()
            for node, compensators in enumerate(simu.tracked_compensator):
                node_residuals[node].extend(np.diff(np.asarray(compensators, dtype=float)))
        residuals = [np.asarray(values, dtype=float) for values in node_residuals]
        return qq_plots(residuals=residuals, **kwargs)


def _format_tick_number(value):
    if value is None:
        return "None"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _looks_like_timestamp_sequence_for_adm4(value):
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or isinstance(value[0], (int, float, np.floating))
    return False


class _ParametricHawkesLearner(_LearnerBase):
    _tick_solvers = ("agd", "bfgs", "gd", "sgd", "svrg")
    _allowed_solvers = {"gd", "agd", "bfgs", "svrg", "sgd", "lbfgs", "lbfgsb", "l-bfgs", "l-bfgs-b"}
    _solver_aliases = {"lbfgs": "bfgs", "lbfgsb": "bfgs", "l-bfgs": "bfgs", "l-bfgs-b": "bfgs"}
    _solver_classes = {"gd": GD, "agd": AGD, "bfgs": BFGS, "sgd": SGD, "svrg": SVRG}
    _solvers_stochastic = {"sgd", "svrg"}
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
        self._solver_obj = None
        self._prox_obj = None
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
            raise ValueError(
                f"``penalty`` must be one of {', '.join(sorted(self._allowed_penalties))}, got {penalty}"
            )
        if solver not in self._allowed_solvers:
            raise ValueError(
                f"``solver`` must be one of {', '.join(self._tick_solvers)}, got {solver}"
            )
        solver = self._solver_aliases.get(solver, solver)
        if C is None or C <= 0:
            raise ValueError(f"``C`` must be positive, got {_format_tick_number(C)}")
        if not 0.0 <= elastic_net_ratio <= 1.0:
            raise ValueError("elastic_net_ratio must be between 0 and 1")
        self._penalty = penalty
        self._solver = solver
        self._C = float(C)
        self._elastic_net_ratio = float(elastic_net_ratio)
        self._step = None
        self._random_state = None
        self._set_initial_step(step)
        self._set_initial_random_state(random_state)
        self.warm_start = bool(warm_start)
        self.coeffs = None
        self.events = None
        self._model_obj = None
        self._solver_obj = self._make_solver_obj()
        self._prox_obj = self._make_prox_obj()
        self.history.print_order = ["n_iter", "obj", "rel_obj", "rel_coeffs"]

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, value):
        raise AttributeError(f"solver is readonly in {self.__class__.__name__}")

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        raise AttributeError(f"penalty is readonly in {self.__class__.__name__}")

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if value is None or value <= 0:
            raise ValueError(f"``C`` must be positive, got {_format_tick_number(value)}")
        if self.penalty == "none":
            warnings.warn(
                f'You cannot set C for penalty "{self.penalty}"',
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._C = float(value)
        if self._prox_obj is not None:
            self._prox_obj.strength = 1.0 / self._C

    @property
    def elastic_net_ratio(self):
        return self._elastic_net_ratio

    @elastic_net_ratio.setter
    def elastic_net_ratio(self, value):
        if self.penalty != "elasticnet":
            warnings.warn(
                f'Penalty "{self.penalty}" has no elastic_net_ratio attribute',
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if not 0.0 <= value <= 1.0:
            raise ValueError("elastic_net_ratio must be between 0 and 1")
        self._elastic_net_ratio = float(value)
        if self._prox_obj is not None:
            self._prox_obj.ratio = self._elastic_net_ratio

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        raise AttributeError(f"random_state is readonly in {self.__class__.__name__}")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = float(value)
        self._sync_solver_attr("tol", self._tol)

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        self._max_iter = int(value)
        self._sync_solver_attr("max_iter", self._max_iter)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)
        self._sync_solver_attr("verbose", self._verbose)

    @property
    def print_every(self):
        return self._print_every

    @print_every.setter
    def print_every(self, value):
        self._print_every = int(value)
        self._sync_solver_attr("print_every", self._print_every)

    @property
    def record_every(self):
        return self._record_every

    @record_every.setter
    def record_every(self, value):
        self._record_every = int(value)
        self._sync_solver_attr("record_every", self._record_every)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if self.solver == "bfgs":
            warnings.warn(
                f'Solver "{self.solver}" has no settable step',
                RuntimeWarning,
                stacklevel=2,
            )
            self._step = None
        else:
            self._step = None if value is None else float(value)
        self._sync_solver_attr("step", self._step)

    def _sync_solver_attr(self, name, value):
        if getattr(self, "_solver_obj", None) is not None:
            setattr(self._solver_obj, name, value)

    def _set_initial_step(self, value):
        if self.solver == "bfgs":
            if value is not None:
                warnings.warn(
                    f'Solver "{self.solver}" has no settable step',
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._step = None
        else:
            self._step = None if value is None else float(value)

    def _set_initial_random_state(self, value):
        if self.solver not in self._solvers_stochastic:
            if value is not None:
                warnings.warn(
                    f'Solver "{self.solver}" has no settable random_state',
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._random_state = None
            return
        if value is not None and value < 0:
            raise ValueError(f"random_state must be positive, got {_format_tick_number(value)}")
        self._random_state = None if value is None else int(value)

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
        if self.solver == "sgd" and self.step is None:
            warnings.warn("SGD step needs to be tuned manually", RuntimeWarning, stacklevel=2)
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
        cls = self._solver_classes[self.solver]
        return cls(
            step=self.step,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            print_every=self.print_every,
            record_every=self.record_every,
            seed=self.random_state,
        )

    def _sync_solver_obj(self):
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
            raise ValueError("You must fit data before getting estimated baseline")
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
            raise ValueError(
                "Parameter gofit (goodness of fit) must be either 'least-squares' or 'likelihood'"
            )
        if gofit == "likelihood" and not isinstance(decays, (int, float, np.floating)):
            decays_arr = np.asarray(decays, dtype=float)
            if not np.allclose(decays_arr, decays_arr.flat[0]):
                raise NotImplementedError(
                    "With 'likelihood' goodness of fit, you must provide a constant decay for all kernels"
                )
        self._gofit = gofit
        self._decays = decays
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
        self._model_obj = self._construct_model_obj()

    @property
    def gofit(self):
        return self._gofit

    @gofit.setter
    def gofit(self, value):
        raise AttributeError(f"gofit is readonly in {self.__class__.__name__}")

    @property
    def decays(self):
        return self._decays

    @decays.setter
    def decays(self, value):
        raise AttributeError(f"decays is readonly in {self.__class__.__name__}")

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
            raise ValueError("You must fit data before getting estimated adjacency")
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
        self._decays = np.asarray(decays, dtype=float)
        self._n_baselines = int(n_baselines)
        self._period_length = period_length
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
        self._model_obj = self._construct_model_obj()

    @property
    def decays(self):
        return self._decays

    @decays.setter
    def decays(self, value):
        raise AttributeError(f"decays is readonly in {self.__class__.__name__}")

    @property
    def n_baselines(self):
        return self._n_baselines

    @n_baselines.setter
    def n_baselines(self, value):
        raise AttributeError(f"n_baselines is readonly in {self.__class__.__name__}")

    @property
    def period_length(self):
        return self._period_length

    @period_length.setter
    def period_length(self, value):
        raise AttributeError(f"period_length is readonly in {self.__class__.__name__}")

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
            raise ValueError("You must fit data before getting estimated baseline")
        raw = self.coeffs[: self.n_nodes * self.n_baselines]
        if self.n_baselines == 1:
            return raw
        return raw.reshape((self.n_nodes, self.n_baselines))

    @property
    def adjacency(self):
        if not self._fitted:
            raise ValueError("You must fit data before getting estimated adjacency")
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


class _HawkesEMPythonBackend:
    """Small compatibility facade for tick's private C++ HawkesEM backend."""

    def __init__(self, owner: "HawkesEM"):
        self.owner = owner
        self._integral_baseline = None
        self._integral_kernel = None

    def get_kernel_support(self):
        return self.owner.kernel_support

    def set_kernel_support(self, val):
        self.owner._set_uniform_kernel_grid(val, self.owner.kernel_size)

    def get_kernel_size(self):
        return self.owner.kernel_size

    def set_kernel_size(self, val):
        self.owner._set_uniform_kernel_grid(self.owner.kernel_support, val)

    def get_kernel_fixed_dt(self):
        return self.owner.kernel_dt

    def set_kernel_dt(self, val):
        self.owner.kernel_dt = val

    def get_kernel_discretization(self):
        return self.owner.kernel_discretization

    def get_kernel_norms(self, flat_kernels):
        kernel = self._reshape_flat_kernels(flat_kernels)
        return np.einsum("ijk,k->ij", kernel, np.diff(self.owner._kernel_discretization))

    def get_kernel_primitives(self, flat_kernels):
        kernel = self._reshape_flat_kernels(flat_kernels)
        dt = np.diff(self.owner._kernel_discretization)
        primitives = np.cumsum(kernel * dt[None, None, :], axis=2)
        return primitives.reshape((self.owner.n_nodes, self.owner.n_nodes * self.owner.kernel_size))

    def loglikelihood(self, baseline, flat_kernels):
        kernel = self._reshape_flat_kernels(flat_kernels)
        return _piecewise_loglik(
            self.owner.data,
            self.owner._end_times,
            np.asarray(baseline, dtype=float),
            kernel,
            self.owner._kernel_discretization,
        )

    def set_buffer_variables_for_integral_of_intensity(self, baseline, flat_kernels):
        self._integral_baseline = np.asarray(baseline, dtype=float)
        self._integral_kernel = self._reshape_flat_kernels(flat_kernels)

    def primitive_of_intensity_at_jump_times(self, realization_node):
        if self._integral_baseline is None or self._integral_kernel is None:
            raise RuntimeError("integral buffers have not been initialized")
        n_nodes = self.owner.n_nodes
        realization_index = int(realization_node) // n_nodes
        node = int(realization_node) % n_nodes
        realization = self.owner.data[realization_index]
        return np.asarray(
            [
                _piecewise_compensator_at(
                    float(t),
                    node,
                    realization,
                    self._integral_baseline,
                    self._integral_kernel,
                    self.owner._kernel_discretization,
                )
                for t in realization[node]
            ],
            dtype=float,
        )

    def _reshape_flat_kernels(self, flat_kernels):
        n_nodes = self.owner.n_nodes
        return np.asarray(flat_kernels, dtype=float).reshape(
            (n_nodes, n_nodes, self.owner.kernel_size)
        )


def _hawkes_em_step_reference(events, sizes, end_times, baseline, kernel, discretization):
    n_realizations, n_nodes, _ = events.shape
    kernel_size = kernel.shape[2]
    support = float(discretization[-1])
    total_time = float(np.sum(end_times))
    next_baseline = np.zeros(n_nodes, dtype=float)
    next_kernel = np.zeros_like(kernel)
    node_counts = np.zeros(n_nodes, dtype=float)
    dt = np.diff(discretization)

    for r in range(n_realizations):
        for v in range(n_nodes):
            node_counts[v] += int(sizes[r, v])
    node_counts = np.maximum(node_counts, 1.0)

    for r in range(n_realizations):
        for u in range(n_nodes):
            for event_index in range(int(sizes[r, u])):
                t = float(events[r, u, event_index])
                contributions = np.zeros((n_nodes, kernel_size), dtype=float)
                intensity = float(baseline[u])
                for v in range(n_nodes):
                    for source_index in range(int(sizes[r, v])):
                        tj = float(events[r, v, source_index])
                        if tj >= t:
                            break
                        lag = float(t - tj)
                        if lag >= support:
                            continue
                        m = int(np.searchsorted(discretization, lag) - 1)
                        if 0 <= m < kernel_size:
                            value = float(kernel[u, v, m])
                            contributions[v, m] += value
                            intensity += value
                if intensity <= 0.0:
                    continue
                next_baseline[u] += float(baseline[u]) / intensity
                next_kernel[u] += contributions / intensity

    out_baseline = np.maximum(next_baseline / max(total_time, 1e-15), 0.0)
    out_kernel = np.empty_like(kernel)
    for u in range(n_nodes):
        for v in range(n_nodes):
            for m in range(kernel_size):
                out_kernel[u, v, m] = max(next_kernel[u, v, m] / (node_counts[v] * dt[m]), 0.0)
    return out_baseline, out_kernel


def _hawkes_em_step_numba_impl(events, sizes, end_times, baseline, kernel, discretization, out_baseline, out_kernel):
    n_realizations, n_nodes, _ = events.shape
    kernel_size = kernel.shape[2]
    support = discretization[discretization.size - 1]
    total_time = 0.0
    for r in range(n_realizations):
        total_time += end_times[r]

    node_counts = np.zeros(n_nodes, dtype=np.float64)
    next_baseline = np.zeros(n_nodes, dtype=np.float64)
    next_kernel = np.zeros((n_nodes, n_nodes, kernel_size), dtype=np.float64)

    for r in range(n_realizations):
        for v in range(n_nodes):
            node_counts[v] += sizes[r, v]
    for v in range(n_nodes):
        if node_counts[v] < 1.0:
            node_counts[v] = 1.0

    for r in range(n_realizations):
        for u in range(n_nodes):
            for event_index in range(sizes[r, u]):
                t = events[r, u, event_index]
                intensity = baseline[u]
                contributions = np.zeros((n_nodes, kernel_size), dtype=np.float64)
                for v in range(n_nodes):
                    for source_index in range(sizes[r, v]):
                        tj = events[r, v, source_index]
                        if tj >= t:
                            break
                        lag = t - tj
                        if lag >= support:
                            continue
                        m = -1
                        for boundary in range(discretization.size):
                            if discretization[boundary] > lag:
                                m = boundary - 1
                                break
                        if m == -1:
                            m = discretization.size - 1
                        if 0 <= m < kernel_size:
                            value = kernel[u, v, m]
                            contributions[v, m] += value
                            intensity += value
                if intensity <= 0.0:
                    continue
                next_baseline[u] += baseline[u] / intensity
                for v in range(n_nodes):
                    for m in range(kernel_size):
                        next_kernel[u, v, m] += contributions[v, m] / intensity

    denom_time = total_time
    if denom_time < 1e-15:
        denom_time = 1e-15
    for u in range(n_nodes):
        value = next_baseline[u] / denom_time
        out_baseline[u] = value if value > 0.0 else 0.0

    for u in range(n_nodes):
        for v in range(n_nodes):
            for m in range(kernel_size):
                dt = discretization[m + 1] - discretization[m]
                value = next_kernel[u, v, m] / (node_counts[v] * dt)
                out_kernel[u, v, m] = value if value > 0.0 else 0.0


_hawkes_em_step_numba = _compile(_hawkes_em_step_numba_impl)


def _hawkes_em_step(events, sizes, end_times, baseline, kernel, discretization):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    end_times = np.ascontiguousarray(np.asarray(end_times, dtype=float))
    baseline = np.ascontiguousarray(np.asarray(baseline, dtype=float))
    kernel = np.ascontiguousarray(np.asarray(kernel, dtype=float))
    discretization = np.ascontiguousarray(np.asarray(discretization, dtype=float))
    if NUMBA_AVAILABLE:
        out_baseline = np.empty_like(baseline)
        out_kernel = np.empty_like(kernel)
        _hawkes_em_step_numba(events, sizes, end_times, baseline, kernel, discretization, out_baseline, out_kernel)
        return out_baseline, out_kernel
    return _hawkes_em_step_reference(events, sizes, end_times, baseline, kernel, discretization)


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
        self._learner = _HawkesEMPythonBackend(self)
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
        if dt > self.kernel_support:
            raise ValueError("kernel_dt must be smaller than kernel_support")
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
        events, sizes, end_times = pack_realizations(self.data, self._end_times)
        self.baseline, self.kernel = _hawkes_em_step(
            events,
            sizes,
            end_times,
            self.baseline,
            self.kernel,
            self._kernel_discretization,
        )

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
        if loss is None:
            kernel = self.kernel if coeffs is None else np.asarray(coeffs, dtype=float)
            loss = -self.score(kernel=kernel)
        return float(loss)

    def get_params(self, deep: bool = True):
        del deep
        return {
            "kernel_support": self.kernel_support,
            "kernel_size": self.kernel_size,
            "kernel_discretization": self.kernel_discretization,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "print_every": self.print_every,
            "record_every": self.record_every,
            "verbose": self.verbose,
            "n_threads": self.n_threads,
        }


def _adm4_compute_weights_reference(events, sizes, end_times, decay):
    n_realizations, n_nodes, max_events = events.shape
    g = np.zeros((n_realizations, n_nodes, max_events, n_nodes), dtype=float)
    map_kernel_integral = np.zeros((n_realizations, n_nodes), dtype=float)

    for r in range(n_realizations):
        end_time = float(end_times[r])
        for u in range(n_nodes):
            for v in range(n_nodes):
                ij = 0
                for k in range(int(sizes[r, u])):
                    t_u_k = float(events[r, u, k])
                    if k > 0:
                        previous_t = float(events[r, u, k - 1])
                        g[r, u, k, v] = g[r, u, k - 1, v] * math.exp(-float(decay) * (t_u_k - previous_t))
                    while ij < int(sizes[r, v]) and float(events[r, v, ij]) < t_u_k:
                        g[r, u, k, v] += float(decay) * math.exp(
                            -float(decay) * (t_u_k - float(events[r, v, ij]))
                        )
                        ij += 1
                    if u == v:
                        map_kernel_integral[r, u] += 1.0 - math.exp(-float(decay) * (end_time - t_u_k))
    return g, map_kernel_integral


def _adm4_em_update_reference(
    g,
    sizes,
    end_times,
    kernel_integral,
    rho,
    mu,
    adjacency,
    z1,
    z2,
    u1,
    u2,
):
    n_realizations, n_nodes, _, _ = g.shape
    next_mu = np.zeros((n_realizations, n_nodes), dtype=float)
    next_C = np.zeros((n_realizations * n_nodes, n_nodes), dtype=float)

    for r in range(n_realizations):
        for node_u in range(n_nodes):
            mu_u = float(mu[node_u])
            adjacency_u = adjacency[node_u]
            next_C_ru = next_C[r * n_nodes + node_u]
            for i in range(int(sizes[r, node_u]) - 1, -1, -1):
                norm = mu_u
                for v in range(n_nodes):
                    norm += float(adjacency_u[v]) * float(g[r, node_u, i, v])
                if norm <= 0.0:
                    continue
                next_mu[r, node_u] += mu_u / norm
                for v in range(n_nodes):
                    next_C_ru[v] += float(adjacency_u[v]) * float(g[r, node_u, i, v]) / norm

    end_times_sum = float(np.sum(end_times))
    for u in range(n_nodes):
        for v in range(n_nodes):
            b_value = float(kernel_integral[v]) + float(rho) * (
                -float(z1[u, v]) + float(u1[u, v]) - float(z2[u, v]) + float(u2[u, v])
            )
            c_value = 0.0
            for r in range(n_realizations):
                c_value += float(next_C[r * n_nodes + u, v])
            adjacency[u, v] = (-b_value + math.sqrt(b_value * b_value + 8.0 * float(rho) * c_value)) / (
                4.0 * float(rho)
            )
        mu[u] = float(np.sum(next_mu[:, u]) / end_times_sum)
    return next_mu, next_C


def _adm4_compute_weights_numba_impl(events, sizes, end_times, decay, g, map_kernel_integral):
    n_realizations, n_nodes, max_events = events.shape
    for r in range(n_realizations):
        for u in range(n_nodes):
            map_kernel_integral[r, u] = 0.0
            for k in range(max_events):
                for v in range(n_nodes):
                    g[r, u, k, v] = 0.0

    for r in range(n_realizations):
        end_time = end_times[r]
        for u in range(n_nodes):
            for v in range(n_nodes):
                ij = 0
                for k in range(sizes[r, u]):
                    t_u_k = events[r, u, k]
                    if k > 0:
                        previous_t = events[r, u, k - 1]
                        g[r, u, k, v] = g[r, u, k - 1, v] * math.exp(-decay * (t_u_k - previous_t))
                    while ij < sizes[r, v] and events[r, v, ij] < t_u_k:
                        g[r, u, k, v] += decay * math.exp(-decay * (t_u_k - events[r, v, ij]))
                        ij += 1
                    if u == v:
                        map_kernel_integral[r, u] += 1.0 - math.exp(-decay * (end_time - t_u_k))


def _adm4_em_update_numba_impl(
    g,
    sizes,
    end_times,
    kernel_integral,
    rho,
    mu,
    adjacency,
    z1,
    z2,
    u1,
    u2,
    next_mu,
    next_C,
):
    n_realizations, n_nodes, _, _ = g.shape
    for r in range(n_realizations):
        for u in range(n_nodes):
            next_mu[r, u] = 0.0
    for row in range(n_realizations * n_nodes):
        for v in range(n_nodes):
            next_C[row, v] = 0.0

    for r in range(n_realizations):
        for node_u in range(n_nodes):
            mu_u = mu[node_u]
            for i in range(sizes[r, node_u] - 1, -1, -1):
                norm = mu_u
                for v in range(n_nodes):
                    norm += adjacency[node_u, v] * g[r, node_u, i, v]
                if norm <= 0.0:
                    continue
                next_mu[r, node_u] += mu_u / norm
                row = r * n_nodes + node_u
                for v in range(n_nodes):
                    next_C[row, v] += adjacency[node_u, v] * g[r, node_u, i, v] / norm

    end_times_sum = 0.0
    for r in range(n_realizations):
        end_times_sum += end_times[r]
    for u in range(n_nodes):
        for v in range(n_nodes):
            b_value = kernel_integral[v] + rho * (-z1[u, v] + u1[u, v] - z2[u, v] + u2[u, v])
            c_value = 0.0
            for r in range(n_realizations):
                c_value += next_C[r * n_nodes + u, v]
            adjacency[u, v] = (-b_value + math.sqrt(b_value * b_value + 8.0 * rho * c_value)) / (4.0 * rho)
        mu_sum = 0.0
        for r in range(n_realizations):
            mu_sum += next_mu[r, u]
        mu[u] = mu_sum / end_times_sum


_adm4_compute_weights_numba = _compile(_adm4_compute_weights_numba_impl)
_adm4_em_update_numba = _compile(_adm4_em_update_numba_impl)


def _adm4_compute_weights(events, sizes, end_times, decay):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    end_times = np.ascontiguousarray(np.asarray(end_times, dtype=float))
    if NUMBA_AVAILABLE:
        g = np.empty((events.shape[0], events.shape[1], events.shape[2], events.shape[1]), dtype=float)
        map_kernel_integral = np.empty((events.shape[0], events.shape[1]), dtype=float)
        _adm4_compute_weights_numba(events, sizes, end_times, float(decay), g, map_kernel_integral)
        return g, map_kernel_integral
    return _adm4_compute_weights_reference(events, sizes, end_times, float(decay))


def _adm4_em_update(g, sizes, end_times, kernel_integral, rho, mu, adjacency, z1, z2, u1, u2):
    g = np.ascontiguousarray(np.asarray(g, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    end_times = np.ascontiguousarray(np.asarray(end_times, dtype=float))
    kernel_integral = np.ascontiguousarray(np.asarray(kernel_integral, dtype=float))
    if NUMBA_AVAILABLE:
        next_mu = np.zeros((g.shape[0], g.shape[1]), dtype=float)
        next_C = np.zeros((g.shape[0] * g.shape[1], g.shape[1]), dtype=float)
        _adm4_em_update_numba(
            g,
            sizes,
            end_times,
            kernel_integral,
            float(rho),
            mu,
            adjacency,
            z1,
            z2,
            u1,
            u2,
            next_mu,
            next_C,
        )
        return next_mu, next_C
    return _adm4_em_update_reference(g, sizes, end_times, kernel_integral, rho, mu, adjacency, z1, z2, u1, u2)


class _ADM4PythonBackend:
    """NumPy port of tick's private C++ HawkesADM4 update backend."""

    def __init__(self, decay, rho, n_threads=1, approx=0):
        self._decay = None
        self._rho = None
        self.n_threads = n_threads
        self.approx = approx
        self.data = None
        self.end_times = None
        self._events_packed = None
        self._event_sizes = None
        self._g_packed = None
        self.n_nodes = 0
        self.n_realizations = 0
        self.weights_computed = False
        self.g = []
        self.kernel_integral = None
        self.next_mu = None
        self.next_C = None
        self.set_decay(decay)
        self.set_rho(rho)

    def set_data(self, data, end_times, n_nodes):
        self.data = data
        self.end_times = np.asarray(end_times, dtype=float)
        self.n_nodes = int(n_nodes)
        self.n_realizations = len(data)
        self._events_packed = None
        self._event_sizes = None
        self._g_packed = None
        self.weights_computed = False

    def get_decay(self):
        return self._decay

    def set_decay(self, decay):
        decay = float(decay)
        if decay <= 0:
            raise ValueError(f"decay must be positive, received {decay:g}")
        self._decay = decay
        self.weights_computed = False

    def get_rho(self):
        return self._rho

    def set_rho(self, rho):
        rho = float(rho)
        if rho <= 0:
            raise ValueError(f"rho (penalty parameter) must be positive, received {rho:g}")
        self._rho = rho

    def compute_weights(self):
        if self.data is None or self.end_times is None:
            raise ValueError("data must be set before computing ADM4 weights")
        events, sizes, _ = pack_realizations(self.data, self.end_times)
        self._events_packed = events
        self._event_sizes = sizes
        self.next_mu = np.zeros((self.n_realizations, self.n_nodes), dtype=float)
        self.next_C = np.zeros((self.n_realizations * self.n_nodes, self.n_nodes), dtype=float)
        self._g_packed, map_kernel_integral = _adm4_compute_weights(
            events,
            sizes,
            self.end_times,
            self.decay,
        )
        self.g = [
            [self._g_packed[r, u, : int(sizes[r, u]), :].copy() for u in range(self.n_nodes)]
            for r in range(self.n_realizations)
        ]
        self.kernel_integral = np.sum(map_kernel_integral, axis=0)
        self.weights_computed = True

    def solve(self, mu, adjacency, z1, z2, u1, u2):
        if not self.weights_computed:
            self.compute_weights()
        mu = np.asarray(mu, dtype=float)
        adjacency = np.asarray(adjacency, dtype=float)
        self._check_shapes(mu, adjacency, z1, z2, u1, u2)

        self.next_mu, self.next_C = _adm4_em_update(
            self._g_packed,
            self._event_sizes,
            self.end_times,
            self.kernel_integral,
            self.rho,
            mu,
            adjacency,
            z1,
            z2,
            u1,
            u2,
        )

    def _check_shapes(self, mu, adjacency, z1, z2, u1, u2):
        if mu.shape != (self.n_nodes,):
            raise ValueError(f"mu argument must be an array of shape ({self.n_nodes},)")
        expected = (self.n_nodes, self.n_nodes)
        for name, value in {
            "adjacency matrix": adjacency,
            "Z1 matrix": z1,
            "Z2 matrix": z2,
            "U1 matrix": u1,
            "U2 matrix": u2,
        }.items():
            if np.asarray(value).shape != expected:
                raise ValueError(f"{name} must be an array of shape {expected}")

    @property
    def decay(self):
        return self._decay

    @property
    def rho(self):
        return self._rho


class HawkesADM4(_LearnerBase):
    """ADM4 learner for exponential Hawkes kernels."""

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
        self._C = None
        self._lasso_nuclear_ratio = None
        self._decay = None
        self._rho = None
        self._model = None
        self._learner = _ADM4PythonBackend(decay, rho, n_threads, approx)
        self.decay = decay
        self.rho = rho
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.warm_start = bool(warm_start)
        self.baseline = None
        self.adjacency = None
        self._prox_l1 = ProxL1(0.0)
        self._prox_nuclear = ProxNuclear(0.0)
        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.history.print_order = ["n_iter", "obj", "rel_obj", "rel_coeffs", "rel_baseline", "rel_adjacency"]

    @property
    def decay(self):
        return self._decay

    @decay.setter
    def decay(self, value):
        self._learner.set_decay(value)
        self._decay = self._learner.get_decay()
        if self._model is not None:
            self._model.decay = self._decay
            if getattr(self._model, "_fitted", False):
                self._model.decays_matrix = self._model._current_decays_matrix()

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._learner.set_rho(value)
        self._rho = self._learner.get_rho()

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if value is None or value <= 0:
            raise ValueError(f"`C` must be positive, got {value}")
        self._C = float(value)
        self._update_prox_strengths()

    @property
    def lasso_nuclear_ratio(self):
        return self._lasso_nuclear_ratio

    @lasso_nuclear_ratio.setter
    def lasso_nuclear_ratio(self, value):
        if value < 0 or value > 1:
            raise ValueError(f"`lasso_nuclear_ratio` must be between 0 and 1, got {value}")
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
        self._set_data(events, end_times=end_times)
        start = self._coerce_start(baseline_start, adjacency_start)
        self.baseline, self.adjacency = self._unpack_coeffs(start)
        self.solve()
        self._fitted = True
        return self

    def _set_data(self, events, end_times=None):
        self._validate_realization_node_counts(events)
        super()._set_data(events, end_times)
        self._model = ModelHawkesExpKernLogLik(self.decay, n_threads=self.n_threads).fit(
            self.data, end_times=self._end_times
        )
        self._learner.set_data(self.data, self._end_times, self._n_nodes)
        self._prox_nuclear.n_rows = self.n_nodes
        return self

    @staticmethod
    def _validate_realization_node_counts(events):
        if events is None or len(events) == 0:
            return
        first = events[0]
        if isinstance(first, np.ndarray) or _looks_like_timestamp_sequence_for_adm4(first):
            return
        expected = len(first)
        for idx, realization in enumerate(events):
            if len(realization) != expected:
                raise RuntimeError(
                    f"All realizations should have {expected} nodes, but realization {idx} has {len(realization)} nodes"
                )

    def solve(self, baseline_start=None, adjacency_start=None):
        if self.data is None:
            raise ValueError("fit data before solving")
        if baseline_start is not None or adjacency_start is not None or self.baseline is None or self.adjacency is None:
            start = self._coerce_start(baseline_start, adjacency_start)
            self.baseline, self.adjacency = self._unpack_coeffs(start)
        self.history.clear()

        z1 = np.zeros_like(self.adjacency)
        z2 = np.zeros_like(self.adjacency)
        u1 = np.zeros_like(self.adjacency)
        u2 = np.zeros_like(self.adjacency)
        max_relative_distance = 1e-1

        for i in range(self.max_iter):
            should_record = self._should_record_iter(i)
            if should_record:
                prev_objective = self.objective(self.coeffs)
                prev_baseline = self.baseline.copy()
                prev_adjacency = self.adjacency.copy()

            for _ in range(self.em_max_iter):
                inner_prev_baseline = self.baseline.copy()
                inner_prev_adjacency = self.adjacency.copy()
                self._learner.solve(self.baseline, self.adjacency, z1, z2, u1, u2)
                inner_rel_baseline = relative_distance(self.baseline, inner_prev_baseline)
                inner_rel_adjacency = relative_distance(self.adjacency, inner_prev_adjacency)
                inner_tol = max_relative_distance * 1e-2 if self.em_tol is None else self.em_tol
                if max(inner_rel_baseline, inner_rel_adjacency) < inner_tol:
                    break

            z1 = self._prox_nuclear.call(np.ravel(self.adjacency + u1), step=1.0 / self.rho).reshape(
                self.n_nodes, self.n_nodes
            )
            z2 = self._prox_l1.call(np.ravel(self.adjacency + u2), step=1.0 / self.rho).reshape(
                self.n_nodes, self.n_nodes
            )
            u1 += self.adjacency - z1
            u2 += self.adjacency - z2

            if should_record:
                objective = self.objective(self.coeffs)
                rel_obj = abs(objective - prev_objective) / max(abs(prev_objective), 1e-300)
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_adjacency = relative_distance(self.adjacency, prev_adjacency)
                max_relative_distance = max(rel_baseline, rel_adjacency)
                self._record(
                    n_iter=i + 1,
                    obj=objective,
                    rel_obj=rel_obj,
                    rel_coeffs=max_relative_distance,
                    rel_baseline=rel_baseline,
                    rel_adjacency=rel_adjacency,
                )
                if max_relative_distance <= self.tol and i > 5:
                    break
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
            adjacency = np.random.uniform(0.5, 0.9, (self.n_nodes, self.n_nodes))
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


def _soft_threshold_scalar(z, alpha):
    return max(abs(float(z)) - float(alpha), 0.0) if z > 0 else -max(abs(float(z)) - float(alpha), 0.0)


def _sumgaussians_compute_weights_reference(events, sizes, end_times, means, std):
    n_realizations, n_nodes, max_events = events.shape
    n_gaussians = means.size
    std_sq = std * std
    norm_constant_gauss = std * math.sqrt(2.0 * math.pi)
    norm_constant_erf = std * math.sqrt(2.0)
    g = np.zeros((n_realizations, n_nodes, max_events, n_nodes * n_gaussians), dtype=float)
    map_kernel_integral = np.zeros((n_realizations, n_nodes * n_gaussians), dtype=float)

    for r in range(n_realizations):
        end_time = float(end_times[r])
        for u in range(n_nodes):
            for k in range(int(sizes[r, u])):
                t_u_k = float(events[r, u, k])
                for v in range(n_nodes):
                    for m in range(n_gaussians):
                        ij = 0
                        total = 0.0
                        while ij < int(sizes[r, v]) and float(events[r, v, ij]) < t_u_k:
                            lag = t_u_k - float(events[r, v, ij]) - float(means[m])
                            total += math.exp(-(lag * lag) / (2.0 * std_sq)) / norm_constant_gauss
                            ij += 1
                        g[r, u, k, v * n_gaussians + m] = total
                    if u == v:
                        for m in range(n_gaussians):
                            map_kernel_integral[r, u * n_gaussians + m] += (
                                0.5 * math.erf((end_time - t_u_k - float(means[m])) / norm_constant_erf)
                                + 0.5 * math.erf(float(means[m]) / norm_constant_erf)
                            )
    return g, map_kernel_integral


def _sumgaussians_em_inner_loop_reference(
    g,
    sizes,
    end_times,
    kernel_integral,
    strength_lasso,
    strength_grouplasso,
    em_max_iter,
    mu,
    amplitudes,
):
    n_realizations, n_nodes, _, n_features = g.shape
    n_gaussians = n_features // n_nodes
    next_mu = np.zeros((n_realizations, n_nodes), dtype=float)
    next_C = np.zeros((n_realizations * n_nodes, n_features), dtype=float)
    end_times_sum = float(np.sum(end_times))

    for _ in range(int(em_max_iter)):
        next_mu.fill(0.0)
        next_C.fill(0.0)
        for r in range(n_realizations):
            for u in range(n_nodes):
                mu_u = float(mu[u])
                amplitudes_u = amplitudes[u]
                next_C_ru = next_C[r * n_nodes + u]
                for i in range(int(sizes[r, u]) - 1, -1, -1):
                    norm = mu_u
                    for idx in range(n_features):
                        norm += float(amplitudes_u[idx]) * float(g[r, u, i, idx])
                    if norm <= 0.0:
                        continue
                    next_mu[r, u] += mu_u / norm
                    for idx in range(n_features):
                        next_C_ru[idx] += float(amplitudes_u[idx]) * float(g[r, u, i, idx]) / norm

        for u in range(n_nodes):
            amplitudes_u = amplitudes[u]
            for v in range(n_nodes):
                norm_group = 0.0
                for m in range(n_gaussians):
                    value = float(amplitudes_u[v * n_gaussians + m])
                    norm_group += value * value
                norm_group = math.sqrt(norm_group)
                a_value = float(strength_grouplasso) / norm_group if norm_group != 0.0 else 0.0
                for m in range(n_gaussians):
                    idx = v * n_gaussians + m
                    b_value = float(kernel_integral[idx]) + float(strength_lasso)
                    next_c_sum = 0.0
                    for r in range(n_realizations):
                        next_c_sum += float(next_C[r * n_nodes + u, idx])
                    c_value = -next_c_sum
                    if a_value != 0.0:
                        sol = (-b_value + math.sqrt(b_value * b_value - 4.0 * a_value * c_value)) / (
                            2.0 * a_value
                        )
                    else:
                        sol = -c_value / b_value
                    amplitudes_u[idx] = sol
            mu[u] = float(np.sum(next_mu[:, u]) / end_times_sum)
    return next_mu, next_C


def _sumgaussians_compute_weights_numba_impl(events, sizes, end_times, means, std, g, map_kernel_integral):
    n_realizations, n_nodes, max_events = events.shape
    n_gaussians = means.size
    std_sq = std * std
    norm_constant_gauss = std * math.sqrt(2.0 * math.pi)
    norm_constant_erf = std * math.sqrt(2.0)

    for r in range(n_realizations):
        for idx in range(n_nodes * n_gaussians):
            map_kernel_integral[r, idx] = 0.0
        for u in range(n_nodes):
            for k in range(max_events):
                for idx in range(n_nodes * n_gaussians):
                    g[r, u, k, idx] = 0.0

    for r in range(n_realizations):
        end_time = end_times[r]
        for u in range(n_nodes):
            for k in range(sizes[r, u]):
                t_u_k = events[r, u, k]
                for v in range(n_nodes):
                    for m in range(n_gaussians):
                        ij = 0
                        total = 0.0
                        while ij < sizes[r, v] and events[r, v, ij] < t_u_k:
                            lag = t_u_k - events[r, v, ij] - means[m]
                            total += math.exp(-(lag * lag) / (2.0 * std_sq)) / norm_constant_gauss
                            ij += 1
                        g[r, u, k, v * n_gaussians + m] = total
                    if u == v:
                        for m in range(n_gaussians):
                            map_kernel_integral[r, u * n_gaussians + m] += (
                                0.5 * math.erf((end_time - t_u_k - means[m]) / norm_constant_erf)
                                + 0.5 * math.erf(means[m] / norm_constant_erf)
                            )


def _sumgaussians_em_inner_loop_numba_impl(
    g,
    sizes,
    end_times,
    kernel_integral,
    strength_lasso,
    strength_grouplasso,
    em_max_iter,
    mu,
    amplitudes,
    next_mu,
    next_C,
):
    n_realizations, n_nodes, _, n_features = g.shape
    n_gaussians = n_features // n_nodes
    end_times_sum = 0.0
    for r in range(n_realizations):
        end_times_sum += end_times[r]

    for _ in range(em_max_iter):
        for r in range(n_realizations):
            for u in range(n_nodes):
                next_mu[r, u] = 0.0
        for row in range(n_realizations * n_nodes):
            for idx in range(n_features):
                next_C[row, idx] = 0.0

        for r in range(n_realizations):
            for u in range(n_nodes):
                mu_u = mu[u]
                for i in range(sizes[r, u] - 1, -1, -1):
                    norm = mu_u
                    for idx in range(n_features):
                        norm += amplitudes[u, idx] * g[r, u, i, idx]
                    if norm <= 0.0:
                        continue
                    next_mu[r, u] += mu_u / norm
                    row = r * n_nodes + u
                    for idx in range(n_features):
                        next_C[row, idx] += amplitudes[u, idx] * g[r, u, i, idx] / norm

        for u in range(n_nodes):
            for v in range(n_nodes):
                norm_group = 0.0
                for m in range(n_gaussians):
                    value = amplitudes[u, v * n_gaussians + m]
                    norm_group += value * value
                norm_group = math.sqrt(norm_group)
                a_value = strength_grouplasso / norm_group if norm_group != 0.0 else 0.0
                for m in range(n_gaussians):
                    idx = v * n_gaussians + m
                    b_value = kernel_integral[idx] + strength_lasso
                    next_c_sum = 0.0
                    for r in range(n_realizations):
                        next_c_sum += next_C[r * n_nodes + u, idx]
                    c_value = -next_c_sum
                    if a_value != 0.0:
                        sol = (-b_value + math.sqrt(b_value * b_value - 4.0 * a_value * c_value)) / (
                            2.0 * a_value
                        )
                    else:
                        sol = -c_value / b_value
                    amplitudes[u, idx] = sol
            mu_sum = 0.0
            for r in range(n_realizations):
                mu_sum += next_mu[r, u]
            mu[u] = mu_sum / end_times_sum


_sumgaussians_compute_weights_numba = _compile(_sumgaussians_compute_weights_numba_impl)
_sumgaussians_em_inner_loop_numba = _compile(_sumgaussians_em_inner_loop_numba_impl)


def _sumgaussians_compute_weights(events, sizes, end_times, means, std):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    end_times = np.ascontiguousarray(np.asarray(end_times, dtype=float))
    means = np.ascontiguousarray(np.asarray(means, dtype=float))
    if NUMBA_AVAILABLE:
        g = np.empty((events.shape[0], events.shape[1], events.shape[2], events.shape[1] * means.size), dtype=float)
        map_kernel_integral = np.empty((events.shape[0], events.shape[1] * means.size), dtype=float)
        _sumgaussians_compute_weights_numba(events, sizes, end_times, means, float(std), g, map_kernel_integral)
        return g, map_kernel_integral
    return _sumgaussians_compute_weights_reference(events, sizes, end_times, means, float(std))


def _sumgaussians_em_inner_loop(
    g,
    sizes,
    end_times,
    kernel_integral,
    strength_lasso,
    strength_grouplasso,
    em_max_iter,
    mu,
    amplitudes,
):
    g = np.ascontiguousarray(np.asarray(g, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    end_times = np.ascontiguousarray(np.asarray(end_times, dtype=float))
    kernel_integral = np.ascontiguousarray(np.asarray(kernel_integral, dtype=float))
    if NUMBA_AVAILABLE:
        next_mu = np.zeros((g.shape[0], g.shape[1]), dtype=float)
        next_C = np.zeros((g.shape[0] * g.shape[1], g.shape[3]), dtype=float)
        _sumgaussians_em_inner_loop_numba(
            g,
            sizes,
            end_times,
            kernel_integral,
            float(strength_lasso),
            float(strength_grouplasso),
            int(em_max_iter),
            mu,
            amplitudes,
            next_mu,
            next_C,
        )
        return next_mu, next_C
    return _sumgaussians_em_inner_loop_reference(
        g,
        sizes,
        end_times,
        kernel_integral,
        strength_lasso,
        strength_grouplasso,
        em_max_iter,
        mu,
        amplitudes,
    )


class _SumGaussiansPythonBackend:
    """NumPy port of tick's private C++ HawkesSumGaussians backend."""

    def __init__(
        self,
        n_gaussians,
        max_mean_gaussian,
        step_size,
        strength_lasso,
        strength_grouplasso,
        em_max_iter,
        n_threads=1,
        approx=0,
    ):
        self.n_threads = n_threads
        self.approx = approx
        self.data = None
        self.end_times = None
        self._events_packed = None
        self._event_sizes = None
        self._g_packed = None
        self.n_nodes = 0
        self.n_realizations = 0
        self.weights_computed = False
        self.set_n_gaussians(n_gaussians)
        self.set_em_max_iter(em_max_iter)
        self.set_max_mean_gaussian(max_mean_gaussian)
        self.set_step_size(step_size)
        self.set_strength_lasso(strength_lasso)
        self.set_strength_grouplasso(strength_grouplasso)

    def set_data(self, data, end_times, n_nodes):
        self.data = data
        self.end_times = np.asarray(end_times, dtype=float)
        self.n_nodes = int(n_nodes)
        self.n_realizations = len(data)
        self._events_packed = None
        self._event_sizes = None
        self._g_packed = None
        self.weights_computed = False

    def get_n_gaussians(self):
        return self.n_gaussians

    def set_n_gaussians(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError(f"n_gaussians must be positive, received {value}")
        self.n_gaussians = value
        self.weights_computed = False

    def get_em_max_iter(self):
        return self.em_max_iter

    def set_em_max_iter(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError(f"em_max_iter must be positive, received {value}")
        self.em_max_iter = value

    def get_max_mean_gaussian(self):
        return self.max_mean_gaussian

    def set_max_mean_gaussian(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"max_mean_gaussian must be positive, received {value:g}")
        self.max_mean_gaussian = value
        self.weights_computed = False

    def get_step_size(self):
        return self.step_size

    def set_step_size(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"step_size must be positive, received {value:g}")
        self.step_size = value

    def get_strength_lasso(self):
        return self.strength_lasso

    def set_strength_lasso(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"strength_lasso must be positive, received {value:g}")
        self.strength_lasso = value

    def get_strength_grouplasso(self):
        return self.strength_grouplasso

    def set_strength_grouplasso(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"strength_grouplasso must be positive, received {value:g}")
        self.strength_grouplasso = value

    @property
    def means_gaussians(self):
        return np.arange(self.n_gaussians, dtype=float) * self.max_mean_gaussian / self.n_gaussians

    @property
    def std_gaussian(self):
        return self.max_mean_gaussian / (self.n_gaussians * math.pi)

    def compute_weights(self):
        if self.data is None or self.end_times is None:
            raise ValueError("data must be set before computing Gaussian weights")
        events, sizes, _ = pack_realizations(self.data, self.end_times)
        self._events_packed = events
        self._event_sizes = sizes
        m_count = self.n_gaussians
        self.next_mu = np.zeros((self.n_realizations, self.n_nodes), dtype=float)
        self.next_C = np.zeros((self.n_realizations * self.n_nodes, self.n_nodes * m_count), dtype=float)
        self.unnormalized_next_C = np.zeros_like(self.next_C)
        self._g_packed, map_kernel_integral = _sumgaussians_compute_weights(
            events,
            sizes,
            self.end_times,
            self.means_gaussians,
            self.std_gaussian,
        )
        self.g = [
            [self._g_packed[r, u, : int(sizes[r, u]), :].copy() for u in range(self.n_nodes)]
            for r in range(self.n_realizations)
        ]
        self.kernel_integral = np.sum(map_kernel_integral, axis=0)
        self.weights_computed = True

    def solve(self, mu, amplitudes):
        if not self.weights_computed:
            self.compute_weights()
        mu = np.asarray(mu, dtype=float)
        amplitudes = np.asarray(amplitudes, dtype=float)
        if mu.shape != (self.n_nodes,):
            raise ValueError(f"mu argument must be an array of shape ({self.n_nodes},)")
        if amplitudes.shape != (self.n_nodes, self.n_nodes * self.n_gaussians):
            raise ValueError(
                f"amplitudes matrix must be an array of shape ({self.n_nodes}, {self.n_nodes * self.n_gaussians})"
            )

        amplitudes_old = amplitudes.copy()
        if NUMBA_AVAILABLE and self._g_packed is not None and self._event_sizes is not None:
            self.next_mu, self.next_C = _sumgaussians_em_inner_loop(
                self._g_packed,
                self._event_sizes,
                self.end_times,
                self.kernel_integral,
                self.strength_lasso,
                self.strength_grouplasso,
                self.em_max_iter,
                mu,
                amplitudes,
            )
            self.unnormalized_next_C = np.zeros_like(self.next_C)
        else:
            for _ in range(self.em_max_iter):
                self.next_C.fill(0.0)
                self.next_mu.fill(0.0)
                self._estimate_all(mu, amplitudes)
                self._update_all(mu, amplitudes)
        self._prox_all(amplitudes, amplitudes_old)

    def _estimate_all(self, mu, amplitudes):
        for r in range(self.n_realizations):
            realization = self.data[r]
            for u in range(self.n_nodes):
                timestamps_u = realization[u]
                g_ru = self.g[r][u]
                mu_u = float(mu[u])
                amplitudes_u = amplitudes[u]
                next_C_ru = self.next_C[r * self.n_nodes + u]
                for i in range(timestamps_u.size - 1, -1, -1):
                    unnormalized = amplitudes_u * g_ru[i]
                    norm = mu_u + float(np.sum(unnormalized))
                    if norm <= 0.0:
                        continue
                    self.next_mu[r, u] += mu_u / norm
                    next_C_ru += unnormalized / norm

    def _update_all(self, mu, amplitudes):
        end_times_sum = float(np.sum(self.end_times))
        m_count = self.n_gaussians
        for u in range(self.n_nodes):
            amplitudes_u = amplitudes[u]
            for v in range(self.n_nodes):
                group = amplitudes_u[v * m_count : (v + 1) * m_count]
                norm_group = float(np.linalg.norm(group))
                a_value = self.strength_grouplasso / norm_group if norm_group != 0.0 else 0.0
                for m in range(m_count):
                    idx = v * m_count + m
                    b_value = self.kernel_integral[idx] + self.strength_lasso
                    c_value = -float(np.sum(self.next_C[u :: self.n_nodes, idx]))
                    if a_value != 0.0:
                        sol = (-b_value + math.sqrt(b_value * b_value - 4.0 * a_value * c_value)) / (2.0 * a_value)
                    else:
                        sol = -c_value / b_value
                    amplitudes_u[idx] = sol
            mu[u] = float(np.sum(self.next_mu[:, u]) / end_times_sum)

    def _prox_all(self, amplitudes, amplitudes_old):
        m_count = self.n_gaussians
        for u in range(self.n_nodes):
            for v in range(self.n_nodes):
                grad_q = np.zeros(m_count, dtype=float)
                for m in range(m_count):
                    idx = v * m_count + m
                    c_value = -float(np.sum(self.next_C[u :: self.n_nodes, idx]))
                    old_value = amplitudes_old[u, idx]
                    grad_q[m] = self.kernel_integral[idx] + c_value / old_value if old_value != 0.0 else self.kernel_integral[idx]
                tmp = np.empty(m_count, dtype=float)
                for m in range(m_count):
                    idx = v * m_count + m
                    tmp[m] = _soft_threshold_scalar(
                        amplitudes[u, idx] - self.step_size * grad_q[m],
                        self.step_size * self.strength_lasso,
                    )
                diff_norm = float(np.linalg.norm(tmp))
                if diff_norm <= self.step_size * self.strength_grouplasso:
                    amplitudes[u, v * m_count : (v + 1) * m_count] = 0.0
                else:
                    scale = max(1.0 - self.step_size * self.strength_grouplasso / diff_norm, 0.0)
                    amplitudes[u, v * m_count : (v + 1) * m_count] = scale * tmp


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
        if C is None or C <= 0:
            raise ValueError(f"`C` must be positive, got {C}")
        if lasso_grouplasso_ratio < 0 or lasso_grouplasso_ratio > 1:
            raise ValueError(
                f"`lasso_grouplasso_ratio` must be between 0 and 1, got {lasso_grouplasso_ratio}"
            )
        self._learner = _SumGaussiansPythonBackend(
            n_gaussians,
            max_mean_gaussian,
            step_size,
            lasso_grouplasso_ratio / C,
            (1.0 - lasso_grouplasso_ratio) / C,
            em_max_iter,
            n_threads,
            approx,
        )
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.baseline = None
        self.amplitudes = None
        self._amplitudes_2d = None
        self.history.print_order = ["n_iter", "rel_baseline", "rel_amplitudes"]

    @property
    def n_gaussians(self):
        return self._learner.get_n_gaussians()

    @n_gaussians.setter
    def n_gaussians(self, value):
        self._learner.set_n_gaussians(value)

    @property
    def max_mean_gaussian(self):
        return self._learner.get_max_mean_gaussian()

    @max_mean_gaussian.setter
    def max_mean_gaussian(self, value):
        self._learner.set_max_mean_gaussian(value)

    @property
    def step_size(self):
        return self._learner.get_step_size()

    @step_size.setter
    def step_size(self, value):
        self._learner.set_step_size(value)

    @property
    def em_max_iter(self):
        return self._learner.get_em_max_iter()

    @em_max_iter.setter
    def em_max_iter(self, value):
        self._learner.set_em_max_iter(value)

    @property
    def C(self):
        return 1.0 / (self.strength_grouplasso + self.strength_lasso)

    @C.setter
    def C(self, val):
        if val is None or val <= 0:
            raise ValueError(f"`C` must be positive, got {val}")
        ratio = self.lasso_grouplasso_ratio
        self.strength_lasso = ratio / val
        self.strength_grouplasso = (1.0 - ratio) / val

    @property
    def lasso_grouplasso_ratio(self):
        ratio = self.strength_lasso / self.strength_grouplasso
        return ratio / (1.0 + ratio)

    @lasso_grouplasso_ratio.setter
    def lasso_grouplasso_ratio(self, val):
        if val is None or val < 0 or val > 1:
            raise ValueError(f"`lasso_grouplasso_ratio` must be between 0 and 1, got {val}")
        C = self.C
        self.strength_lasso = float(val) / C
        self.strength_grouplasso = (1.0 - float(val)) / C

    @property
    def strength_lasso(self):
        return self._learner.get_strength_lasso()

    @strength_lasso.setter
    def strength_lasso(self, val):
        self._learner.set_strength_lasso(val)

    @property
    def strength_grouplasso(self):
        return self._learner.get_strength_grouplasso()

    @strength_grouplasso.setter
    def strength_grouplasso(self, val):
        self._learner.set_strength_grouplasso(val)

    @property
    def means_gaussians(self):
        return self._learner.means_gaussians

    @property
    def std_gaussian(self):
        return self._learner.std_gaussian

    def fit(self, events, end_times=None, baseline_start=None, amplitudes_start=None):
        self._set_data(events, end_times)
        self.solve(baseline_start=baseline_start, amplitudes_start=amplitudes_start)
        self._fitted = True
        return self

    def _set_data(self, events, end_times=None):
        HawkesADM4._validate_realization_node_counts(events)
        super()._set_data(events, end_times)
        self._learner.set_data(self.data, self._end_times, self._n_nodes)
        return self

    def solve(self, baseline_start=None, amplitudes_start=None):
        if self.data is None:
            raise ValueError("fit data before solving")
        if baseline_start is None:
            baseline_start = np.ones(self.n_nodes)
        else:
            baseline_start = np.asarray(baseline_start, dtype=float)
            if baseline_start.shape != (self.n_nodes,):
                raise ValueError(f"baseline_start has shape {baseline_start.shape}, expected {(self.n_nodes,)}")
        self.baseline = baseline_start.copy()

        if amplitudes_start is None:
            amplitudes_start = np.random.uniform(0.5, 0.9, (self.n_nodes, self.n_nodes, self.n_gaussians))
        else:
            amplitudes_start = np.asarray(amplitudes_start, dtype=float)
            if amplitudes_start.shape != (self.n_nodes, self.n_nodes, self.n_gaussians):
                raise ValueError(
                    f"amplitudes_start has shape {amplitudes_start.shape} but should have shape "
                    f"{(self.n_nodes, self.n_nodes, self.n_gaussians)}"
                )
        self.amplitudes = amplitudes_start.copy()
        self._amplitudes_2d = self.amplitudes.reshape((self.n_nodes, self.n_nodes * self.n_gaussians))

        max_relative_distance = 1e-1
        for i in range(self.max_iter):
            if self._should_record_iter(i):
                prev_baseline = self.baseline.copy()
                prev_amplitudes = self.amplitudes.copy()
                inner_prev_baseline = self.baseline.copy()
                inner_prev_amplitudes = self.amplitudes.copy()

            self._learner.solve(self.baseline, self._amplitudes_2d)

            if self._should_record_iter(i):
                inner_rel_baseline = relative_distance(self.baseline, inner_prev_baseline)
                inner_rel_amplitudes = relative_distance(self.amplitudes, inner_prev_amplitudes)
                inner_tol = max_relative_distance * 1e-2 if self.em_tol is None else self.em_tol
                if max(inner_rel_baseline, inner_rel_amplitudes) < inner_tol:
                    break
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_amplitudes = relative_distance(self.amplitudes, prev_amplitudes)
                max_relative_distance = max(rel_baseline, rel_amplitudes)
                self._record(n_iter=i + 1, rel_baseline=rel_baseline, rel_amplitudes=rel_amplitudes)
                if max_relative_distance <= self.tol and i > 5:
                    break
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
        amplitudes = self.amplitudes if coeffs is None else np.asarray(coeffs, dtype=float)
        if loss is None:
            loss = self._surrogate_negative_loglik(amplitudes)
        l1 = self.strength_lasso * float(np.sum(np.abs(amplitudes)))
        group = self.strength_grouplasso * float(np.sum(np.linalg.norm(amplitudes, axis=2)))
        return float(loss + l1 + group)

    def _surrogate_negative_loglik(self, amplitudes):
        if not self._fitted:
            raise ValueError("fit must be called first")
        discretization = np.linspace(0.0, self.max_mean_gaussian, self.n_gaussians + 1)
        midpoints = (discretization[:-1] + discretization[1:]) / 2.0
        kernel = np.empty((self.n_nodes, self.n_nodes, self.n_gaussians), dtype=float)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                sigma = self.std_gaussian
                values = np.zeros_like(midpoints)
                for amplitude, mean in zip(amplitudes[i, j], self.means_gaussians):
                    values += amplitude * np.exp(-0.5 * ((midpoints - mean) / sigma) ** 2) / (
                        sigma * np.sqrt(2.0 * np.pi)
                    )
                kernel[i, j] = values
        return -_piecewise_loglik(self.data, self._end_times, self.baseline, kernel, discretization)


def _basis_compute_r_reference(u_realization, end_time, kernel_dt, basis_kernels, basis_primitives, out):
    n_basis, kernel_size = basis_kernels.shape
    for timestamp in u_realization:
        m0 = int(math.floor((float(end_time) - float(timestamp)) / kernel_dt))
        if m0 < 0:
            continue
        for d in range(n_basis):
            if m0 >= kernel_size:
                out[d] += basis_primitives[d, kernel_size - 1]
            elif m0 > 0:
                out[d] += (
                    basis_primitives[d, m0 - 1]
                    + (float(end_time) - float(timestamp) - m0 * kernel_dt) * basis_kernels[d, m0]
                )
            else:
                out[d] += (float(end_time) - float(timestamp) - m0 * kernel_dt) * basis_kernels[d, m0]


def _basis_compute_C_reference(u_realization, end_time, kernel_dt, basis_kernels, amplitude_sums, out):
    if u_realization.size == 0:
        return
    n_basis, kernel_size = basis_kernels.shape
    i = u_realization.size - 1
    for m in range(kernel_size):
        threshold = float(end_time) - m * kernel_dt
        while i >= 0 and u_realization[i] > threshold:
            i -= 1
        if i < 0:
            break
        for d in range(n_basis):
            if basis_kernels[d, m] != 0.0:
                out[d, m] += amplitude_sums[d] * (i + 1) / basis_kernels[d, m]


def _basis_compute_mu_q_D_reference(
    u_index,
    realization,
    kernel_dt,
    basis_kernels,
    amplitudes_u,
    baseline_u,
    qvd,
    Ddm,
):
    n_nodes = len(realization)
    n_basis, kernel_size = basis_kernels.shape
    u_timestamps = realization[u_index]
    v_indices = [timestamps.size for timestamps in realization]
    qvd_temp = np.zeros((n_nodes, n_basis), dtype=float)
    Ddm_temp = np.zeros((n_basis, kernel_size), dtype=float)
    mu_out = 0.0

    for i in range(u_timestamps.size - 1, -1, -1):
        norm = 0.0
        qvd_temp.fill(0.0)
        Ddm_temp.fill(0.0)
        t_i = float(u_timestamps[i])

        for v_index in range(n_nodes):
            v_timestamps = realization[v_index]
            if v_timestamps.size == 0:
                continue
            while True:
                if v_indices[v_index] == 0:
                    break
                if v_indices[v_index] < v_timestamps.size and t_i >= v_timestamps[v_indices[v_index]]:
                    break
                v_indices[v_index] -= 1
            if t_i < v_timestamps[v_indices[v_index]]:
                continue

            for j in range(v_indices[v_index], -1, -1):
                t_j = float(v_timestamps[j])
                if u_index == v_index and i == j:
                    norm += baseline_u
                else:
                    m = int(math.floor((t_i - t_j) / kernel_dt))
                    if m >= kernel_size:
                        break
                    values = amplitudes_u[v_index] * basis_kernels[:, m]
                    qvd_temp[v_index] += values
                    Ddm_temp[:, m] += values
                    norm += float(np.sum(values))

        if norm <= 0.0:
            continue
        mu_out += baseline_u / norm
        Ddm += Ddm_temp / (norm * kernel_dt)
        qvd += amplitudes_u * qvd_temp / norm

    return mu_out


def _basis_compute_gdm_reference(alpha, kernel_dt, basis_kernel, Cdm, Ddm, tol, max_iter):
    basis_kernel.fill(0.0)
    max_rel_error = 0.0
    kernel_size = basis_kernel.size

    for n_iter in range(int(max_iter)):
        max_rel_error = -1.0
        for m in range(kernel_size):
            left = 0.0 if m == 0 else basis_kernel[m - 1]
            right = 0.0 if m == kernel_size - 1 else basis_kernel[m + 1]
            a = 4.0 * alpha / (kernel_dt * kernel_dt) + Cdm[m]
            b = -2.0 * alpha * (right + left) / (kernel_dt * kernel_dt)
            c = -Ddm[m]
            discriminant = b * b - 4.0 * a * c
            sol = (-b + math.sqrt(max(discriminant, 0.0))) / (2.0 * a)
            if n_iter != 0:
                rel_error = sol if basis_kernel[m] == 0.0 else (sol - basis_kernel[m]) / basis_kernel[m]
                max_rel_error = max(rel_error, max_rel_error)
            basis_kernel[m] = sol
        if n_iter > 0 and max_rel_error < tol:
            break

    return max_rel_error


def _basis_compute_r_numba_impl(u_realization, end_time, kernel_dt, basis_kernels, basis_primitives, out):
    n_basis, kernel_size = basis_kernels.shape
    for k in range(u_realization.size):
        timestamp = u_realization[k]
        m0 = int(math.floor((end_time - timestamp) / kernel_dt))
        if m0 < 0:
            continue
        for d in range(n_basis):
            if m0 >= kernel_size:
                out[d] += basis_primitives[d, kernel_size - 1]
            elif m0 > 0:
                out[d] += basis_primitives[d, m0 - 1] + (end_time - timestamp - m0 * kernel_dt) * basis_kernels[d, m0]
            else:
                out[d] += (end_time - timestamp - m0 * kernel_dt) * basis_kernels[d, m0]


def _basis_compute_C_numba_impl(u_realization, end_time, kernel_dt, basis_kernels, amplitude_sums, out):
    if u_realization.size == 0:
        return
    n_basis, kernel_size = basis_kernels.shape
    i = u_realization.size - 1
    for m in range(kernel_size):
        threshold = end_time - m * kernel_dt
        while i >= 0 and u_realization[i] > threshold:
            i -= 1
        if i < 0:
            break
        for d in range(n_basis):
            if basis_kernels[d, m] != 0.0:
                out[d, m] += amplitude_sums[d] * (i + 1) / basis_kernels[d, m]


def _basis_compute_mu_q_D_numba_impl(
    u_index,
    events,
    sizes,
    kernel_dt,
    basis_kernels,
    amplitudes_u,
    baseline_u,
    qvd,
    Ddm,
):
    n_nodes = sizes.size
    n_basis, kernel_size = basis_kernels.shape
    u_size = sizes[u_index]
    v_indices = np.empty(n_nodes, dtype=np.int64)
    for v_index in range(n_nodes):
        v_indices[v_index] = sizes[v_index]
    qvd_temp = np.zeros((n_nodes, n_basis), dtype=np.float64)
    Ddm_temp = np.zeros((n_basis, kernel_size), dtype=np.float64)
    mu_out = 0.0

    for i in range(u_size - 1, -1, -1):
        norm = 0.0
        for v_index in range(n_nodes):
            for d in range(n_basis):
                qvd_temp[v_index, d] = 0.0
        for d in range(n_basis):
            for m in range(kernel_size):
                Ddm_temp[d, m] = 0.0
        t_i = events[u_index, i]

        for v_index in range(n_nodes):
            v_size = sizes[v_index]
            if v_size == 0:
                continue
            while True:
                if v_indices[v_index] == 0:
                    break
                if v_indices[v_index] < v_size and t_i >= events[v_index, v_indices[v_index]]:
                    break
                v_indices[v_index] -= 1
            if t_i < events[v_index, v_indices[v_index]]:
                continue

            for j in range(v_indices[v_index], -1, -1):
                t_j = events[v_index, j]
                if u_index == v_index and i == j:
                    norm += baseline_u
                else:
                    m = int(math.floor((t_i - t_j) / kernel_dt))
                    if m >= kernel_size:
                        break
                    for d in range(n_basis):
                        value = amplitudes_u[v_index, d] * basis_kernels[d, m]
                        qvd_temp[v_index, d] += value
                        Ddm_temp[d, m] += value
                        norm += value

        if norm <= 0.0:
            continue
        mu_out += baseline_u / norm
        scale_d = 1.0 / (norm * kernel_dt)
        for d in range(n_basis):
            for m in range(kernel_size):
                Ddm[d, m] += Ddm_temp[d, m] * scale_d
        for v_index in range(n_nodes):
            for d in range(n_basis):
                qvd[v_index, d] += amplitudes_u[v_index, d] * qvd_temp[v_index, d] / norm

    return mu_out


def _basis_compute_gdm_numba_impl(alpha, kernel_dt, basis_kernel, Cdm, Ddm, tol, max_iter):
    for m in range(basis_kernel.size):
        basis_kernel[m] = 0.0
    max_rel_error = 0.0
    kernel_size = basis_kernel.size

    for n_iter in range(int(max_iter)):
        max_rel_error = -1.0
        for m in range(kernel_size):
            left = 0.0 if m == 0 else basis_kernel[m - 1]
            right = 0.0 if m == kernel_size - 1 else basis_kernel[m + 1]
            a = 4.0 * alpha / (kernel_dt * kernel_dt) + Cdm[m]
            b = -2.0 * alpha * (right + left) / (kernel_dt * kernel_dt)
            c = -Ddm[m]
            discriminant = b * b - 4.0 * a * c
            sol = (-b + math.sqrt(max(discriminant, 0.0))) / (2.0 * a)
            if n_iter != 0:
                rel_error = sol if basis_kernel[m] == 0.0 else (sol - basis_kernel[m]) / basis_kernel[m]
                max_rel_error = max(rel_error, max_rel_error)
            basis_kernel[m] = sol
        if n_iter > 0 and max_rel_error < tol:
            break

    return max_rel_error


_basis_compute_r_numba = _compile(_basis_compute_r_numba_impl)
_basis_compute_C_numba = _compile(_basis_compute_C_numba_impl)
_basis_compute_mu_q_D_numba = _compile(_basis_compute_mu_q_D_numba_impl)
_basis_compute_gdm_numba = _compile(_basis_compute_gdm_numba_impl)


def _basis_compute_r(u_realization, end_time, kernel_dt, basis_kernels, basis_primitives, out):
    if NUMBA_AVAILABLE:
        _basis_compute_r_numba(u_realization, float(end_time), float(kernel_dt), basis_kernels, basis_primitives, out)
        return
    _basis_compute_r_reference(u_realization, end_time, kernel_dt, basis_kernels, basis_primitives, out)


def _basis_compute_C(u_realization, end_time, kernel_dt, basis_kernels, amplitude_sums, out):
    if NUMBA_AVAILABLE:
        _basis_compute_C_numba(
            u_realization,
            float(end_time),
            float(kernel_dt),
            basis_kernels,
            amplitude_sums,
            out,
        )
        return
    _basis_compute_C_reference(u_realization, end_time, kernel_dt, basis_kernels, amplitude_sums, out)


def _basis_compute_mu_q_D(
    u_index,
    realization,
    kernel_dt,
    basis_kernels,
    amplitudes_u,
    baseline_u,
    qvd,
    Ddm,
):
    if NUMBA_AVAILABLE:
        events, sizes = pack_realization(realization)
        return _basis_compute_mu_q_D_numba(
            int(u_index),
            events,
            sizes,
            float(kernel_dt),
            basis_kernels,
            amplitudes_u,
            float(baseline_u),
            qvd,
            Ddm,
        )
    return _basis_compute_mu_q_D_reference(
        u_index,
        realization,
        kernel_dt,
        basis_kernels,
        amplitudes_u,
        baseline_u,
        qvd,
        Ddm,
    )


def _basis_compute_gdm(alpha, kernel_dt, basis_kernel, Cdm, Ddm, tol, max_iter):
    if NUMBA_AVAILABLE:
        return _basis_compute_gdm_numba(
            float(alpha),
            float(kernel_dt),
            basis_kernel,
            Cdm,
            Ddm,
            float(tol),
            int(max_iter),
        )
    return _basis_compute_gdm_reference(alpha, kernel_dt, basis_kernel, Cdm, Ddm, tol, max_iter)


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
        self._amplitudes_2d = None

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
        self.solve(
            baseline_start=baseline_start,
            amplitudes_start=amplitudes_start,
            basis_kernels_start=basis_kernels_start,
        )
        self._fitted = True
        return self

    def _set_data(self, events, end_times=None):
        HawkesADM4._validate_realization_node_counts(events)
        super()._set_data(events, end_times)
        return self

    def solve(self, baseline_start=None, amplitudes_start=None, basis_kernels_start=None):
        if self.data is None:
            raise ValueError("fit data before solving")
        if self.n_basis in (None, 0):
            self.n_basis = self.n_nodes

        if baseline_start is None:
            self.baseline = np.ones(self.n_nodes, dtype=float)
        else:
            self.baseline = np.asarray(baseline_start, dtype=float).copy()

        if amplitudes_start is None:
            self.amplitudes = np.random.uniform(0.5, 0.9, size=(self.n_nodes, self.n_nodes, self.n_basis))
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

        self._amplitudes_2d = self.amplitudes.reshape((self.n_nodes, self.n_nodes * self.n_basis))

        for i in range(self.max_iter):
            if self._should_record_iter(i):
                prev_baseline = self.baseline.copy()
                prev_amplitudes = self.amplitudes.copy()
                prev_basis_kernels = self.basis_kernels.copy()

            rel_ode = self._solve_one_iteration()

            if self._should_record_iter(i):
                rel_baseline = relative_distance(self.baseline, prev_baseline)
                rel_amplitudes = relative_distance(self.amplitudes, prev_amplitudes)
                rel_basis_kernels = relative_distance(self.basis_kernels, prev_basis_kernels)
                converged = max(rel_baseline, rel_amplitudes, rel_basis_kernels) <= self.tol
                self._record(
                    n_iter=i + 1,
                    rel_baseline=rel_baseline,
                    rel_amplitudes=rel_amplitudes,
                    rel_basis_kernels=rel_basis_kernels,
                    rel_ode=rel_ode,
                    force=(i + 1 == self.max_iter) or converged,
                )
                if converged:
                    break
        return self

    def _solve_one_iteration(self):
        n_nodes = self.n_nodes
        n_basis = self.n_basis
        kernel_size = self.kernel_size
        kernel_dt = self.kernel_dt
        alpha = 1.0 / self.C

        basis_primitives = np.cumsum(self.basis_kernels, axis=1) * kernel_dt
        amplitude_sums = np.sum(self.amplitudes, axis=0)
        rud = np.zeros((n_nodes, n_basis), dtype=float)
        quvd = np.zeros((n_nodes, n_nodes, n_basis), dtype=float)
        Dudm = np.zeros((n_nodes, n_basis, kernel_size), dtype=float)
        Cudm = np.zeros((n_nodes, n_basis, kernel_size), dtype=float)
        end_times_sum = float(np.sum(self._end_times))

        for u in range(n_nodes):
            mu_out = 0.0
            for realization, end_time in zip(self.data, self._end_times):
                _basis_compute_r(realization[u], end_time, kernel_dt, self.basis_kernels, basis_primitives, rud[u])
                _basis_compute_C(
                    realization[u],
                    end_time,
                    kernel_dt,
                    self.basis_kernels,
                    amplitude_sums[u],
                    Cudm[u],
                )
                mu_out += _basis_compute_mu_q_D(
                    u,
                    realization,
                    kernel_dt,
                    self.basis_kernels,
                    self.amplitudes[u],
                    float(self.baseline[u]),
                    quvd[u],
                    Dudm[u],
                )
            self.baseline[u] = mu_out / end_times_sum

        for u in range(n_nodes):
            for v in range(n_nodes):
                for d in range(n_basis):
                    denominator = rud[v, d] + 2.0 * alpha
                    self.amplitudes[u, v, d] = math.sqrt(quvd[u, v, d] / denominator) if denominator > 0 else 0.0

        Csum = np.sum(Cudm, axis=0)
        Dsum = np.sum(Dudm, axis=0)
        rel_ode = 0.0
        for d in range(n_basis):
            rel_ode = max(
                rel_ode,
                _basis_compute_gdm(
                    alpha,
                    kernel_dt,
                    self.basis_kernels[d],
                    Csum[d],
                    Dsum[d],
                    self.ode_tol,
                    self.ode_max_iter,
                ),
            )
        return rel_ode

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
        amplitudes = self.amplitudes if coeffs is None else np.asarray(coeffs, dtype=float)
        if loss is None:
            kernel = np.tensordot(amplitudes, self.basis_kernels, axes=(2, 0))
            loss = -_piecewise_loglik(
                self.data,
                self._end_times,
                self.baseline,
                kernel,
                self._kernel_discretization,
            )
        amplitude_penalty = float(np.sum(amplitudes ** 2)) / self.C
        if self.basis_kernels.shape[1] > 1:
            smoothness = float(np.sum(np.diff(self.basis_kernels, axis=1) ** 2)) / self.C
        else:
            smoothness = 0.0
        return float(loss + amplitude_penalty + smoothness)


def _conditional_law_pack_realization(realization, marks):
    events, sizes = pack_realization(realization)
    packed_marks = np.zeros_like(events)
    for node, node_marks in enumerate(marks):
        node_marks = np.ascontiguousarray(np.asarray(node_marks, dtype=float))
        if node_marks.size != int(sizes[node]):
            raise ValueError("marks must match timestamps length")
        if node_marks.size:
            packed_marks[node, : node_marks.size] = node_marks
    return events, sizes, packed_marks


def _conditional_law_pack_signal_pairs(pairs):
    max_size = max((len(pair[0]) for pair in pairs), default=0)
    x_values = np.zeros((len(pairs), max_size), dtype=float)
    y_values = np.zeros((len(pairs), max_size), dtype=float)
    sizes = np.zeros(len(pairs), dtype=np.int64)
    for index, pair in enumerate(pairs):
        x = np.ascontiguousarray(np.asarray(pair[0], dtype=float))
        y = np.ascontiguousarray(np.asarray(pair[1], dtype=float))
        if x.shape != y.shape:
            raise ValueError("packed signal x/y arrays must have matching shapes")
        sizes[index] = x.size
        if x.size:
            x_values[index, : x.size] = x
            y_values[index, : y.size] = y
    return x_values, y_values, sizes


def _conditional_law_pack_linear_system(
    ijl2index,
    index2ijl,
    mark_probabilities,
    int_claw,
    IG,
    IG2,
):
    n_nodes = len(ijl2index)
    max_marks = max((len(mark_probabilities[node]) for node in range(n_nodes)), default=0)
    ijl_to_index = np.full((n_nodes, n_nodes, max_marks), -1, dtype=np.int64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            for l, index in enumerate(ijl2index[i][j]):
                ijl_to_index[i, j, l] = int(index)

    index2ijl_array = np.ascontiguousarray(np.asarray(index2ijl, dtype=np.int64))
    mark_probabilities_array = np.zeros((n_nodes, max_marks), dtype=float)
    for node, probabilities in enumerate(mark_probabilities):
        probabilities = np.ascontiguousarray(np.asarray(probabilities, dtype=float))
        if probabilities.size:
            mark_probabilities_array[node, : probabilities.size] = probabilities

    int_x, int_y, int_sizes = _conditional_law_pack_signal_pairs(int_claw)
    ig_x, ig_y, ig_sizes = _conditional_law_pack_signal_pairs(IG)
    ig2_x, ig2_y, ig2_sizes = _conditional_law_pack_signal_pairs(IG2)
    return (
        index2ijl_array,
        ijl_to_index,
        mark_probabilities_array,
        int_x,
        int_y,
        int_sizes,
        ig_x,
        ig_y,
        ig_sizes,
        ig2_x,
        ig2_y,
        ig2_sizes,
    )


def _conditional_law_point_process_cond_law_reference(
    events,
    sizes,
    marks,
    y_node,
    z_node,
    lags,
    zmin,
    zmax,
    y_T,
    y_lambda,
):
    n_lags = lags.size - 1
    res_x = np.zeros(n_lags, dtype=float)
    res_y = np.zeros(n_lags, dtype=float)
    lag_max = float(lags[n_lags])
    tab_y_index = np.zeros(n_lags, dtype=np.int64)
    n_terms = 0
    y_index = 0
    y_size = int(sizes[y_node])
    z_size = int(sizes[z_node])
    for z_index in range(z_size):
        z_derivative = float(marks[z_node, z_index])
        if z_index > 0:
            z_derivative -= float(marks[z_node, z_index - 1])
        if zmin < zmax and z_index > 0 and (zmin > z_derivative or z_derivative > zmax):
            continue
        z_t = float(events[z_node, z_index])
        if z_t + lag_max >= y_T:
            break
        n_terms += 1
        while y_index < y_size and float(events[y_node, y_index]) < z_t:
            y_index += 1
        if y_index >= y_size:
            break
        y_index_lag_delta = y_index
        for k in range(n_lags):
            lag = float(lags[k])
            y_index_lag = y_index_lag_delta
            while y_index_lag < y_size and float(events[y_node, y_index_lag]) <= z_t + lag:
                y_index_lag += 1
            if y_index_lag >= y_size:
                y_index_lag_delta = y_size - 1
                tab_y_index[k] = y_index_lag_delta
                continue
            ytlag = 0 if y_index_lag == 0 else y_index_lag - 1
            y_index_lag_delta = max(y_index_lag, int(tab_y_index[k]))
            while y_index_lag_delta < y_size and float(events[y_node, y_index_lag_delta]) <= z_t + lags[k + 1]:
                y_index_lag_delta += 1
            if y_index_lag_delta >= y_size:
                y_index_lag_delta = y_size - 1
                if y_index_lag == y_size - 1:
                    tab_y_index[k] = y_index_lag_delta
                    continue
            tab_y_index[k] = y_index_lag_delta
            ytlagdelta = 0 if y_index_lag_delta == 0 else y_index_lag_delta - 1
            res_y[k] += ytlagdelta - ytlag
    for k in range(n_lags):
        if n_terms != 0:
            res_y[k] /= n_terms
        res_y[k] /= lags[k + 1] - lags[k]
        res_y[k] -= y_lambda
        res_x[k] = (lags[k + 1] + lags[k]) / 2.0
    return res_x, res_y


def _conditional_law_lin0_reference(x_values, y_values, size, t):
    if t >= x_values[size - 1]:
        return 0.0
    index = int(np.searchsorted(x_values[:size], t))
    if index == size - 1:
        return float(y_values[index])
    if abs(float(x_values[index]) - t) < abs(float(x_values[index + 1]) - t):
        return float(y_values[index])
    return float(y_values[index + 1])


def _conditional_law_linc_reference(x_values, y_values, size, t):
    if t >= x_values[size - 1]:
        return float(y_values[size - 1])
    index = int(np.searchsorted(x_values[:size], t))
    if index == size - 1:
        return float(y_values[index])
    if abs(float(x_values[index]) - t) < abs(float(x_values[index + 1]) - t):
        return float(y_values[index])
    return float(y_values[index + 1])


def _conditional_law_G_reference(i, j, l, t, ijl_to_index, int_x, int_y, int_sizes):
    index = int(ijl_to_index[i, j, l])
    return _conditional_law_lin0_reference(int_x[index], int_y[index], int(int_sizes[index]), float(t))


def _conditional_law_DIG_reference(i, j, l, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes):
    index = int(ijl_to_index[i, j, l])
    size = int(ig_sizes[index])
    return _conditional_law_linc_reference(ig_x[index], ig_y[index], size, float(t2)) - _conditional_law_linc_reference(
        ig_x[index],
        ig_y[index],
        size,
        float(t1),
    )


def _conditional_law_compute_V_reference(i, n_index, n_quad, index_first, index_last, index2ijl, ijl_to_index, quad_x, int_x, int_y, int_sizes):
    V = np.zeros((n_index * n_quad, 1), dtype=float)
    for index in range(index_first, index_last + 1):
        _, j, l = index2ijl[index]
        for n in range(n_quad):
            row = (index - index_first) * n_quad + n
            V[row, 0] = _conditional_law_G_reference(i, j, l, float(quad_x[n]), ijl_to_index, int_x, int_y, int_sizes)
    return V


def _conditional_law_compute_M_reference(
    n_index,
    n_quad,
    index_first,
    index_last,
    method_code,
    index2ijl,
    ijl_to_index,
    mean_intensity,
    mark_probabilities,
    quad_x,
    quad_w,
    int_x,
    int_y,
    int_sizes,
    ig_x,
    ig_y,
    ig_sizes,
    ig2_x,
    ig2_y,
    ig2_sizes,
):
    M = np.zeros((n_index * n_quad, n_index * n_quad), dtype=float)
    for index in range(index_first, index_last + 1):
        _, j, l = index2ijl[index]
        for index1 in range(index_first, index_last + 1):
            _, j1, l1 = index2ijl[index1]
            fact = float(mean_intensity[j1] / mean_intensity[j])
            mark_probability = float(mark_probabilities[j1, l1])
            for n in range(n_quad):
                for n1 in range(n_quad):
                    if method_code in {0, 1}:
                        if n > n1:
                            x = mark_probability * float(quad_w[n1]) * _conditional_law_G_reference(
                                j1,
                                j,
                                l,
                                float(quad_x[n] - quad_x[n1]),
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                        elif n < n1:
                            x = fact * mark_probability * float(quad_w[n1]) * _conditional_law_G_reference(
                                j,
                                j1,
                                l1,
                                float(quad_x[n1] - quad_x[n]),
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                        elif method_code == 1:
                            x = 0.0
                        else:
                            x1 = mark_probability * float(quad_w[n1]) * _conditional_law_G_reference(
                                j1,
                                j,
                                l,
                                0.0,
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                            x2 = fact * mark_probability * float(quad_w[n1]) * _conditional_law_G_reference(
                                j,
                                j1,
                                l1,
                                0.0,
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                            x = (x1 + x2) / 2.0
                        if method_code == 1:
                            row = (index - index_first) * n_quad + n
                            col = (index1 - index_first) * n_quad + n
                            M[row, col] -= x
                    else:
                        x = _conditional_law_compute_M_log_lin_value_reference(
                            n,
                            n1,
                            n_quad,
                            j,
                            l,
                            j1,
                            l1,
                            fact,
                            mark_probability,
                            quad_x,
                            quad_w,
                            ijl_to_index,
                            ig_x,
                            ig_y,
                            ig_sizes,
                            ig2_x,
                            ig2_y,
                            ig2_sizes,
                        )
                    if l == l1 and j == j1 and n == n1:
                        x += 1.0
                    row = (index - index_first) * n_quad + n
                    col = (index1 - index_first) * n_quad + n1
                    M[row, col] += x
    return M


def _conditional_law_compute_M_log_lin_value_reference(
    n,
    n1,
    n_quad,
    j,
    l,
    j1,
    l1,
    fact,
    mark_probability,
    quad_x,
    quad_w,
    ijl_to_index,
    ig_x,
    ig_y,
    ig_sizes,
    ig2_x,
    ig2_y,
    ig2_sizes,
):
    def dig(j_lower, j_greater, l_greater, t1, t2):
        return _conditional_law_DIG_reference(j_lower, j_greater, l_greater, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)

    def dig2(j_lower, j_greater, l_greater, t1, t2):
        return _conditional_law_DIG_reference(j_lower, j_greater, l_greater, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)

    def ratio_dig(n_q):
        return (float(quad_x[n]) - float(quad_x[n_q])) / float(quad_w[n_q])

    def ratio_dig2(n_q):
        return 1.0 / float(quad_w[n_q])

    def greater_args(n_q):
        return (
            j1,
            j,
            l,
            float(quad_x[n] - quad_x[n_q] - quad_w[n_q]),
            float(quad_x[n] - quad_x[n_q]),
        )

    def lower_args(n_q):
        return (
            j,
            j1,
            l1,
            float(quad_x[n_q] - quad_x[n]),
            float(quad_x[n_q] - quad_x[n] + quad_w[n_q]),
        )

    x = 0.0
    if n > n1:
        x += mark_probability * dig(*greater_args(n1))
        if n1 < n_quad - 1:
            x -= ratio_dig(n1) * mark_probability * dig(*greater_args(n1))
            x += ratio_dig2(n1) * mark_probability * dig2(*greater_args(n1))
        if n1 > 0:
            x += ratio_dig(n1 - 1) * mark_probability * dig(*greater_args(n1 - 1))
            x -= ratio_dig2(n1 - 1) * mark_probability * dig2(*greater_args(n1 - 1))
    elif n < n1:
        x += fact * mark_probability * dig(*lower_args(n1))
        if n1 < n_quad - 1:
            x -= fact * ratio_dig(n1) * mark_probability * dig(*lower_args(n1))
            x -= fact * ratio_dig2(n1) * mark_probability * dig2(*lower_args(n1))
        if n1 > 0:
            x += fact * ratio_dig(n1 - 1) * mark_probability * dig(*lower_args(n1 - 1))
            x += fact * ratio_dig2(n1 - 1) * mark_probability * dig2(*lower_args(n1 - 1))
    else:
        x += fact * mark_probability * dig(*lower_args(n1))
        if n1 < n_quad - 1:
            x -= fact * ratio_dig(n1) * mark_probability * dig(*lower_args(n1))
            x -= fact * ratio_dig2(n1) * mark_probability * dig2(*lower_args(n1))
        if n1 > 0:
            x += ratio_dig(n1 - 1) * mark_probability * dig(*greater_args(n1 - 1))
            x -= ratio_dig2(n1 - 1) * mark_probability * dig2(*greater_args(n1 - 1))
    return x


def _conditional_law_point_process_cond_law_numba_impl(
    events,
    sizes,
    marks,
    y_node,
    z_node,
    lags,
    zmin,
    zmax,
    y_T,
    y_lambda,
    res_x,
    res_y,
):
    n_lags = lags.size - 1
    lag_max = lags[n_lags]
    tab_y_index = np.zeros(n_lags, dtype=np.int64)
    n_terms = 0
    y_index = 0
    y_size = sizes[y_node]
    z_size = sizes[z_node]
    for k in range(n_lags):
        res_x[k] = 0.0
        res_y[k] = 0.0
    for z_index in range(z_size):
        z_derivative = marks[z_node, z_index]
        if z_index > 0:
            z_derivative -= marks[z_node, z_index - 1]
        if zmin < zmax and z_index > 0 and (zmin > z_derivative or z_derivative > zmax):
            continue
        z_t = events[z_node, z_index]
        if z_t + lag_max >= y_T:
            break
        n_terms += 1
        while y_index < y_size and events[y_node, y_index] < z_t:
            y_index += 1
        if y_index >= y_size:
            break
        y_index_lag_delta = y_index
        for k in range(n_lags):
            lag = lags[k]
            y_index_lag = y_index_lag_delta
            while y_index_lag < y_size and events[y_node, y_index_lag] <= z_t + lag:
                y_index_lag += 1
            if y_index_lag >= y_size:
                y_index_lag_delta = y_size - 1
                tab_y_index[k] = y_index_lag_delta
                continue
            ytlag = 0
            if y_index_lag != 0:
                ytlag = y_index_lag - 1
            y_index_lag_delta = y_index_lag
            if y_index_lag_delta < tab_y_index[k]:
                y_index_lag_delta = tab_y_index[k]
            while y_index_lag_delta < y_size and events[y_node, y_index_lag_delta] <= z_t + lags[k + 1]:
                y_index_lag_delta += 1
            if y_index_lag_delta >= y_size:
                y_index_lag_delta = y_size - 1
                if y_index_lag == y_size - 1:
                    tab_y_index[k] = y_index_lag_delta
                    continue
            tab_y_index[k] = y_index_lag_delta
            ytlagdelta = 0
            if y_index_lag_delta != 0:
                ytlagdelta = y_index_lag_delta - 1
            res_y[k] += ytlagdelta - ytlag
    for k in range(n_lags):
        if n_terms != 0:
            res_y[k] /= n_terms
        res_y[k] /= lags[k + 1] - lags[k]
        res_y[k] -= y_lambda
        res_x[k] = (lags[k + 1] + lags[k]) / 2.0


def _conditional_law_lin0_numba(x_values, y_values, size, t):
    if t >= x_values[size - 1]:
        return 0.0
    index = 0
    while index < size and x_values[index] < t:
        index += 1
    if index == size - 1:
        return y_values[index]
    if abs(x_values[index] - t) < abs(x_values[index + 1] - t):
        return y_values[index]
    return y_values[index + 1]


def _conditional_law_linc_numba(x_values, y_values, size, t):
    if t >= x_values[size - 1]:
        return y_values[size - 1]
    index = 0
    while index < size and x_values[index] < t:
        index += 1
    if index == size - 1:
        return y_values[index]
    if abs(x_values[index] - t) < abs(x_values[index + 1] - t):
        return y_values[index]
    return y_values[index + 1]


def _conditional_law_G_numba(i, j, l, t, ijl_to_index, int_x, int_y, int_sizes):
    index = ijl_to_index[i, j, l]
    return _conditional_law_lin0_numba(int_x[index], int_y[index], int_sizes[index], t)


def _conditional_law_DIG_numba(i, j, l, t1, t2, ijl_to_index, x_values, y_values, sizes):
    index = ijl_to_index[i, j, l]
    size = sizes[index]
    return _conditional_law_linc_numba(x_values[index], y_values[index], size, t2) - _conditional_law_linc_numba(
        x_values[index],
        y_values[index],
        size,
        t1,
    )


def _conditional_law_compute_V_numba_impl(i, n_index, n_quad, index_first, index_last, index2ijl, ijl_to_index, quad_x, int_x, int_y, int_sizes, V):
    for row in range(n_index * n_quad):
        V[row, 0] = 0.0
    for index in range(index_first, index_last + 1):
        j = index2ijl[index, 1]
        l = index2ijl[index, 2]
        for n in range(n_quad):
            row = (index - index_first) * n_quad + n
            V[row, 0] = _conditional_law_G_numba(i, j, l, quad_x[n], ijl_to_index, int_x, int_y, int_sizes)


def _conditional_law_compute_M_log_lin_value_numba(
    n,
    n1,
    n_quad,
    j,
    l,
    j1,
    l1,
    fact,
    mark_probability,
    quad_x,
    quad_w,
    ijl_to_index,
    ig_x,
    ig_y,
    ig_sizes,
    ig2_x,
    ig2_y,
    ig2_sizes,
):
    x = 0.0
    if n > n1:
        t1 = quad_x[n] - quad_x[n1] - quad_w[n1]
        t2 = quad_x[n] - quad_x[n1]
        dig = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
        dig2 = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
        x += mark_probability * dig
        if n1 < n_quad - 1:
            ratio = (quad_x[n] - quad_x[n1]) / quad_w[n1]
            x -= ratio * mark_probability * dig
            x += (1.0 / quad_w[n1]) * mark_probability * dig2
        if n1 > 0:
            t1 = quad_x[n] - quad_x[n1 - 1] - quad_w[n1 - 1]
            t2 = quad_x[n] - quad_x[n1 - 1]
            dig = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
            dig2 = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
            ratio = (quad_x[n] - quad_x[n1 - 1]) / quad_w[n1 - 1]
            x += ratio * mark_probability * dig
            x -= (1.0 / quad_w[n1 - 1]) * mark_probability * dig2
    elif n < n1:
        t1 = quad_x[n1] - quad_x[n]
        t2 = quad_x[n1] - quad_x[n] + quad_w[n1]
        dig = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
        dig2 = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
        x += fact * mark_probability * dig
        if n1 < n_quad - 1:
            ratio = (quad_x[n] - quad_x[n1]) / quad_w[n1]
            x -= fact * ratio * mark_probability * dig
            x -= fact * (1.0 / quad_w[n1]) * mark_probability * dig2
        if n1 > 0:
            t1 = quad_x[n1 - 1] - quad_x[n]
            t2 = quad_x[n1 - 1] - quad_x[n] + quad_w[n1 - 1]
            dig = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
            dig2 = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
            ratio = (quad_x[n] - quad_x[n1 - 1]) / quad_w[n1 - 1]
            x += fact * ratio * mark_probability * dig
            x += fact * (1.0 / quad_w[n1 - 1]) * mark_probability * dig2
    else:
        t1 = quad_x[n1] - quad_x[n]
        t2 = quad_x[n1] - quad_x[n] + quad_w[n1]
        dig = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
        dig2 = _conditional_law_DIG_numba(j, j1, l1, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
        x += fact * mark_probability * dig
        if n1 < n_quad - 1:
            ratio = (quad_x[n] - quad_x[n1]) / quad_w[n1]
            x -= fact * ratio * mark_probability * dig
            x -= fact * (1.0 / quad_w[n1]) * mark_probability * dig2
        if n1 > 0:
            t1 = quad_x[n] - quad_x[n1 - 1] - quad_w[n1 - 1]
            t2 = quad_x[n] - quad_x[n1 - 1]
            dig = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig_x, ig_y, ig_sizes)
            dig2 = _conditional_law_DIG_numba(j1, j, l, t1, t2, ijl_to_index, ig2_x, ig2_y, ig2_sizes)
            ratio = (quad_x[n] - quad_x[n1 - 1]) / quad_w[n1 - 1]
            x += ratio * mark_probability * dig
            x -= (1.0 / quad_w[n1 - 1]) * mark_probability * dig2
    return x


def _conditional_law_compute_M_numba_impl(
    n_index,
    n_quad,
    index_first,
    index_last,
    method_code,
    index2ijl,
    ijl_to_index,
    mean_intensity,
    mark_probabilities,
    quad_x,
    quad_w,
    int_x,
    int_y,
    int_sizes,
    ig_x,
    ig_y,
    ig_sizes,
    ig2_x,
    ig2_y,
    ig2_sizes,
    M,
):
    for row in range(n_index * n_quad):
        for col in range(n_index * n_quad):
            M[row, col] = 0.0
    for index in range(index_first, index_last + 1):
        j = index2ijl[index, 1]
        l = index2ijl[index, 2]
        for index1 in range(index_first, index_last + 1):
            j1 = index2ijl[index1, 1]
            l1 = index2ijl[index1, 2]
            fact = mean_intensity[j1] / mean_intensity[j]
            mark_probability = mark_probabilities[j1, l1]
            for n in range(n_quad):
                for n1 in range(n_quad):
                    if method_code == 0 or method_code == 1:
                        if n > n1:
                            x = mark_probability * quad_w[n1] * _conditional_law_G_numba(
                                j1,
                                j,
                                l,
                                quad_x[n] - quad_x[n1],
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                        elif n < n1:
                            x = fact * mark_probability * quad_w[n1] * _conditional_law_G_numba(
                                j,
                                j1,
                                l1,
                                quad_x[n1] - quad_x[n],
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                        elif method_code == 1:
                            x = 0.0
                        else:
                            x1 = mark_probability * quad_w[n1] * _conditional_law_G_numba(
                                j1,
                                j,
                                l,
                                0.0,
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                            x2 = fact * mark_probability * quad_w[n1] * _conditional_law_G_numba(
                                j,
                                j1,
                                l1,
                                0.0,
                                ijl_to_index,
                                int_x,
                                int_y,
                                int_sizes,
                            )
                            x = (x1 + x2) / 2.0
                        if method_code == 1:
                            row = (index - index_first) * n_quad + n
                            col = (index1 - index_first) * n_quad + n
                            M[row, col] -= x
                    else:
                        x = _conditional_law_compute_M_log_lin_value_numba(
                            n,
                            n1,
                            n_quad,
                            j,
                            l,
                            j1,
                            l1,
                            fact,
                            mark_probability,
                            quad_x,
                            quad_w,
                            ijl_to_index,
                            ig_x,
                            ig_y,
                            ig_sizes,
                            ig2_x,
                            ig2_y,
                            ig2_sizes,
                        )
                    if l == l1 and j == j1 and n == n1:
                        x += 1.0
                    row = (index - index_first) * n_quad + n
                    col = (index1 - index_first) * n_quad + n1
                    M[row, col] += x


_conditional_law_lin0_numba = _compile(_conditional_law_lin0_numba)
_conditional_law_linc_numba = _compile(_conditional_law_linc_numba)
_conditional_law_G_numba = _compile(_conditional_law_G_numba)
_conditional_law_DIG_numba = _compile(_conditional_law_DIG_numba)
_conditional_law_compute_M_log_lin_value_numba = _compile(_conditional_law_compute_M_log_lin_value_numba)
_conditional_law_point_process_cond_law_numba = _compile(_conditional_law_point_process_cond_law_numba_impl)
_conditional_law_compute_V_numba = _compile(_conditional_law_compute_V_numba_impl)
_conditional_law_compute_M_numba = _compile(_conditional_law_compute_M_numba_impl)


def _conditional_law_method_code(method):
    if method == "gauss":
        return 0
    if method == "gauss-":
        return 1
    return 2


def _conditional_law_point_process_cond_law(events, sizes, marks, y_node, z_node, lags, zmin, zmax, y_T, y_lambda):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    marks = np.ascontiguousarray(np.asarray(marks, dtype=float))
    lags = np.ascontiguousarray(np.asarray(lags, dtype=float))
    if NUMBA_AVAILABLE:
        res_x = np.zeros(lags.size - 1, dtype=float)
        res_y = np.zeros(lags.size - 1, dtype=float)
        _conditional_law_point_process_cond_law_numba(
            events,
            sizes,
            marks,
            int(y_node),
            int(z_node),
            lags,
            float(zmin),
            float(zmax),
            float(y_T),
            float(y_lambda),
            res_x,
            res_y,
        )
        return res_x, res_y
    return _conditional_law_point_process_cond_law_reference(
        events,
        sizes,
        marks,
        int(y_node),
        int(z_node),
        lags,
        zmin,
        zmax,
        y_T,
        y_lambda,
    )


def _conditional_law_compute_V(i, n_index, n_quad, index_first, index_last, index2ijl, ijl_to_index, quad_x, int_x, int_y, int_sizes):
    index2ijl = np.ascontiguousarray(np.asarray(index2ijl, dtype=np.int64))
    ijl_to_index = np.ascontiguousarray(np.asarray(ijl_to_index, dtype=np.int64))
    quad_x = np.ascontiguousarray(np.asarray(quad_x, dtype=float))
    int_x = np.ascontiguousarray(np.asarray(int_x, dtype=float))
    int_y = np.ascontiguousarray(np.asarray(int_y, dtype=float))
    int_sizes = np.ascontiguousarray(np.asarray(int_sizes, dtype=np.int64))
    if NUMBA_AVAILABLE:
        V = np.empty((int(n_index) * int(n_quad), 1), dtype=float)
        _conditional_law_compute_V_numba(
            int(i),
            int(n_index),
            int(n_quad),
            int(index_first),
            int(index_last),
            index2ijl,
            ijl_to_index,
            quad_x,
            int_x,
            int_y,
            int_sizes,
            V,
        )
        return V
    return _conditional_law_compute_V_reference(
        int(i),
        int(n_index),
        int(n_quad),
        int(index_first),
        int(index_last),
        index2ijl,
        ijl_to_index,
        quad_x,
        int_x,
        int_y,
        int_sizes,
    )


def _conditional_law_compute_M(
    n_index,
    n_quad,
    index_first,
    index_last,
    method,
    index2ijl,
    ijl_to_index,
    mean_intensity,
    mark_probabilities,
    quad_x,
    quad_w,
    int_x,
    int_y,
    int_sizes,
    ig_x,
    ig_y,
    ig_sizes,
    ig2_x,
    ig2_y,
    ig2_sizes,
):
    method_code = _conditional_law_method_code(method)
    index2ijl = np.ascontiguousarray(np.asarray(index2ijl, dtype=np.int64))
    ijl_to_index = np.ascontiguousarray(np.asarray(ijl_to_index, dtype=np.int64))
    mean_intensity = np.ascontiguousarray(np.asarray(mean_intensity, dtype=float))
    mark_probabilities = np.ascontiguousarray(np.asarray(mark_probabilities, dtype=float))
    quad_x = np.ascontiguousarray(np.asarray(quad_x, dtype=float))
    quad_w = np.ascontiguousarray(np.asarray(quad_w, dtype=float))
    int_x = np.ascontiguousarray(np.asarray(int_x, dtype=float))
    int_y = np.ascontiguousarray(np.asarray(int_y, dtype=float))
    int_sizes = np.ascontiguousarray(np.asarray(int_sizes, dtype=np.int64))
    ig_x = np.ascontiguousarray(np.asarray(ig_x, dtype=float))
    ig_y = np.ascontiguousarray(np.asarray(ig_y, dtype=float))
    ig_sizes = np.ascontiguousarray(np.asarray(ig_sizes, dtype=np.int64))
    ig2_x = np.ascontiguousarray(np.asarray(ig2_x, dtype=float))
    ig2_y = np.ascontiguousarray(np.asarray(ig2_y, dtype=float))
    ig2_sizes = np.ascontiguousarray(np.asarray(ig2_sizes, dtype=np.int64))
    if NUMBA_AVAILABLE:
        M = np.empty((int(n_index) * int(n_quad), int(n_index) * int(n_quad)), dtype=float)
        _conditional_law_compute_M_numba(
            int(n_index),
            int(n_quad),
            int(index_first),
            int(index_last),
            int(method_code),
            index2ijl,
            ijl_to_index,
            mean_intensity,
            mark_probabilities,
            quad_x,
            quad_w,
            int_x,
            int_y,
            int_sizes,
            ig_x,
            ig_y,
            ig_sizes,
            ig2_x,
            ig2_y,
            ig2_sizes,
            M,
        )
        return M
    return _conditional_law_compute_M_reference(
        int(n_index),
        int(n_quad),
        int(index_first),
        int(index_last),
        method_code,
        index2ijl,
        ijl_to_index,
        mean_intensity,
        mark_probabilities,
        quad_x,
        quad_w,
        int_x,
        int_y,
        int_sizes,
        ig_x,
        ig_y,
        ig_sizes,
        ig2_x,
        ig2_y,
        ig2_sizes,
    )


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
        self._packed_realizations = []
        self._conditional_linear_pack = None
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
        self._packed_realizations = []
        self._conditional_linear_pack = None
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
            self._packed_realizations = []
            self._conditional_linear_pack = None
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
        self._packed_realizations.append(_conditional_law_pack_realization(stored, stored_marks))
        self._conditional_linear_pack = None

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
        self._compute_tick_conditional_laws()
        self._set_tick_quadrature()
        self._compute_ints_claw()
        index_first = 0
        self._phi_ijl = []
        self._norm_ijl = []
        self.kernels = []
        self.kernels_norms = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            index_last = index_first
            for index_last in range(index_first, self._n_index):
                i1, _, _ = self._index2ijl[index_last]
                if i1 != i:
                    index_last -= 1
                    break
            n_index = index_last - index_first + 1
            V = self._compute_V(i, n_index, self.n_quad, index_first, index_last)
            M = self._compute_M(n_index, self.n_quad, index_first, index_last, self.quad_method)
            res = scipy.linalg.solve(M, V, assume_a="gen")
            self._estimate_kernels_and_norms(i, index_first, index_last, res, self.n_quad, self.quad_method)
            index_first = index_last + 1
        self._apply_final_kernel_symmetries()
        self._estimate_baseline()
        self._estimate_mark_functions()

    def _compute_tick_conditional_laws(self):
        if len(self._packed_realizations) != len(self.data):
            self._packed_realizations = [
                _conditional_law_pack_realization(realization, marks)
                for realization, marks in zip(self.data, self._marks)
            ]
        n_marks = [len(intervals) for intervals in self.marked_components]
        self._mark_probabilities_N = [np.zeros(n, dtype=float) for n in n_marks]
        self._mark_probabilities = [np.zeros(n, dtype=float) for n in n_marks]
        self._mark_min = np.full(self.n_nodes, np.finfo(float).max, dtype=float)
        self._mark_max = np.full(self.n_nodes, np.finfo(float).min, dtype=float)
        self._lam_N = np.zeros(self.n_nodes, dtype=float)
        self._lam_T = np.zeros(self.n_nodes, dtype=float)
        self.mean_intensity = np.zeros(self.n_nodes, dtype=float)
        self._n_events = np.zeros((2, self.n_nodes), dtype=float)
        claw_sums = [np.zeros(len(self._lags) - 1, dtype=float) for _ in range(self._n_index)]
        n_realizations = len(self.data)
        self._claw_X = np.zeros(len(self._lags) - 1, dtype=float)

        for r, (realization, marks, end_time) in enumerate(zip(self.data, self._marks, self._end_times)):
            end_time = float(end_time)
            packed_events, packed_sizes, packed_marks = self._packed_realizations[r]
            for i in range(self.n_nodes):
                if realization[i].size == 0:
                    continue
                derivatives = np.hstack((marks[i][0], np.diff(marks[i])))
                self._mark_min[i] = min(self._mark_min[i], float(np.min(derivatives)))
                self._mark_max[i] = max(self._mark_max[i], float(np.max(derivatives)))
                for l, interval in enumerate(self.marked_components[i]):
                    self._mark_probabilities_N[i][l] += np.sum(
                        (derivatives >= interval[0]) & (derivatives < interval[1])
                    )

            for i in range(self.n_nodes):
                if realization[i].size == 0:
                    continue
                self._lam_N[i] += realization[i].size
                self._lam_T[i] += end_time

            for i in range(self.n_nodes):
                good = np.sum(realization[i] <= end_time - self._lags[-1])
                bad = realization[i].size - good
                self._n_events[0, i] += good
                self._n_events[1, i] += bad

            for index, (i, j, l) in enumerate(self._index2ijl):
                lambda_i = realization[i].size / end_time if end_time > 0 else 0.0
                claw_x, claw_y = _conditional_law_point_process_cond_law(
                    packed_events,
                    packed_sizes,
                    packed_marks,
                    i,
                    j,
                    self._lags,
                    self.marked_components[j][l][0],
                    self.marked_components[j][l][1],
                    end_time,
                    lambda_i,
                )
                self._claw_X = claw_x
                claw_sums[index] += claw_y

        self._claw = [claw / max(n_realizations, 1) for claw in claw_sums]
        for i in range(self.n_nodes):
            total = float(np.sum(self._mark_probabilities_N[i]))
            if total > 0:
                self._mark_probabilities[i] = self._mark_probabilities_N[i] / total
            elif self._mark_probabilities[i].size:
                self._mark_probabilities[i][0] = 1.0
            if self._lam_T[i] > 0:
                self.mean_intensity[i] = self._lam_N[i] / self._lam_T[i]
            if self._mark_min[i] == np.finfo(float).max:
                self._mark_min[i] = 1.0
                self._mark_max[i] = 1.0

        self._claw1 = self._aggregate_unmarked_claws()
        self._apply_tick_symmetries()

    @staticmethod
    def _point_process_cond_law(y_time, z_time, z_mark, lags, zmin, zmax, y_T, y_lambda):
        if z_time.size != z_mark.size:
            raise ValueError("z_time and z_mark should have the same size")
        events, sizes, marks = _conditional_law_pack_realization(
            [y_time, z_time],
            [np.zeros_like(y_time, dtype=float), z_mark],
        )
        return _conditional_law_point_process_cond_law(events, sizes, marks, 0, 1, lags, zmin, zmax, y_T, y_lambda)

    def _aggregate_unmarked_claws(self):
        claw1 = []
        for i in range(self.n_nodes):
            row = []
            for j in range(self.n_nodes):
                index = self._ijl2index[i][j][0]
                values = np.copy(self._claw[index]) * self._mark_probabilities[j][0]
                for l in range(1, len(self._ijl2index[i][j])):
                    index = self._ijl2index[i][j][l]
                    values += self._claw[index] * self._mark_probabilities[j][l]
                row.append(values)
            claw1.append(row)
        return claw1

    def _apply_tick_symmetries(self):
        for i, j in self.symmetries1d:
            value = (self.mean_intensity[i] + self.mean_intensity[j]) / 2.0
            self.mean_intensity[i] = value
            self.mean_intensity[j] = value
            value = (self._mark_min[i] + self._mark_min[j]) / 2.0
            self._mark_min[i] = value
            self._mark_min[j] = value
            value = (self._mark_max[i] + self._mark_max[j]) / 2.0
            self._mark_max[i] = value
            self._mark_max[j] = value
            if self.marked_components[i] != self.marked_components[j]:
                continue
            for l in range(len(self.marked_components[i])):
                value = (self._mark_probabilities_N[i][l] + self._mark_probabilities_N[j][l]) / 2.0
                self._mark_probabilities_N[i][l] = value
                self._mark_probabilities_N[j][l] = value
                value = (self._mark_probabilities[i][l] + self._mark_probabilities[j][l]) / 2.0
                self._mark_probabilities[i][l] = value
                self._mark_probabilities[j][l] = value

        for (i1, j1), (i2, j2) in self.symmetries2d:
            value = (self._claw1[i1][j1] + self._claw1[i2][j2]) / 2.0
            self._claw1[i1][j1] = value
            self._claw1[i2][j2] = value
            if self.marked_components[j1] != self.marked_components[j2]:
                continue
            for l in range(len(self.marked_components[j1])):
                index1 = self._ijl2index[i1][j1][l]
                index2 = self._ijl2index[i2][j2][l]
                value = (self._claw[index1] + self._claw[index2]) / 2.0
                self._claw[index1] = value
                self._claw[index2] = value

    def _set_tick_quadrature(self):
        if self.quad_method in {"gauss", "gauss-"}:
            self._quad_x, self._quad_w = leggauss(self.n_quad)
            self._quad_x = self.max_support * (self._quad_x + 1.0) / 2.0
            self._quad_w *= self.max_support / 2.0
        elif self.quad_method == "log":
            logstep = (np.log(self.max_support) - np.log(self.min_support) + 1.0) / self.n_quad
            x1 = np.arange(0.0, self.min_support, self.min_support * logstep)
            x2 = np.exp(np.arange(np.log(self.min_support), np.log(self.max_support), logstep))
            self._quad_x = np.append(x1, x2)
            self._quad_w = self._quad_x[1:] - self._quad_x[:-1]
            self._quad_w = np.append(self._quad_w, self._quad_w[-1])
            self.n_quad = len(self._quad_x)
            self._quad_x = np.asarray(self._quad_x, dtype=float)
            self._quad_w = np.asarray(self._quad_w, dtype=float)
        elif self.quad_method == "lin":
            self._quad_x = np.arange(0.0, self.max_support, self.max_support / self.n_quad)
            self._quad_w = self._quad_x[1:] - self._quad_x[:-1]
            self._quad_w = np.append(self._quad_w, self._quad_w[-1])
            self.n_quad = len(self._quad_x)
            self._quad_x = np.asarray(self._quad_x, dtype=float)
            self._quad_w = np.asarray(self._quad_w, dtype=float)

    def _compute_ints_claw(self):
        self._int_claw = [None] * self._n_index
        for index in range(self._n_index):
            xe = self._claw_X
            ye = self._claw[index]
            xs2 = np.array([(a - b) for a, b in product(self._quad_x, repeat=2)], dtype=float)
            xs2 = np.append(xe, xs2)
            xs2 = np.append(self._quad_x, xs2)
            xs2 = np.array(np.unique(xs2))
            xs2 = np.sort(xs2)
            xs2 = xs2[xs2 >= 0.0]
            ys2 = np.zeros(len(xs2), dtype=float)
            j = 0
            for i in range(1, len(xe)):
                while j < len(xs2) and xs2[j] < xe[i]:
                    ys2[j] = ye[i - 1] + (ye[i] - ye[i - 1]) * (xs2[j] - xe[i - 1]) / (xe[i] - xe[i - 1])
                    j += 1
            self._int_claw[index] = (xs2, ys2)
        self._IG = []
        self._IG2 = []
        for i in range(self._n_index):
            xc = self._int_claw[i][0]
            yc = self._int_claw[i][1]
            self._IG.append((xc, np.append(np.array(0.0), np.cumsum(np.diff(xc) * (yc[:-1] + yc[1:]) / 2.0))))
            self._IG2.append(
                (
                    xc,
                    np.append(
                        np.array(0.0),
                        np.cumsum(
                            (yc[:-1] + yc[1:]) / 2.0 * np.diff(xc) * xc[:-1]
                            + np.diff(xc) * np.diff(xc) / 3.0 * np.diff(yc)
                            + np.diff(xc) * np.diff(xc) / 2.0 * yc[:-1]
                        ),
                    ),
                )
            )
        self._conditional_linear_pack = _conditional_law_pack_linear_system(
            self._ijl2index,
            self._index2ijl,
            self._mark_probabilities,
            self._int_claw,
            self._IG,
            self._IG2,
        )

    @staticmethod
    def _lin0(sig, t):
        x, y = sig
        if t >= x[-1]:
            return 0.0
        index = np.searchsorted(x, t)
        if index == len(y) - 1:
            return y[index]
        if abs(x[index] - t) < abs(x[index + 1] - t):
            return y[index]
        return y[index + 1]

    @staticmethod
    def _linc(sig, t):
        x, y = sig
        if t >= x[-1]:
            return y[-1]
        index = np.searchsorted(x, t)
        if abs(x[index] - t) < abs(x[index + 1] - t):
            return y[index]
        return y[index + 1]

    def _G(self, i, j, l, t):
        if t < 0:
            warnings.warn("G(): should not be called for t < 0", UserWarning, stacklevel=2)
        index = self._ijl2index[i][j][l]
        return self._lin0(self._int_claw[index], t)

    def _DIG(self, i, j, l, t1, t2):
        if t1 >= t2:
            warnings.warn("t2>t1 wrong in DIG", UserWarning, stacklevel=2)
        index = self._ijl2index[i][j][l]
        return self._linc(self._IG[index], t2) - self._linc(self._IG[index], t1)

    def _DIG2(self, i, j, l, t1, t2):
        if t1 >= t2:
            warnings.warn("t2>t1 wrong in DIG2", UserWarning, stacklevel=2)
        index = self._ijl2index[i][j][l]
        return self._linc(self._IG2[index], t2) - self._linc(self._IG2[index], t1)

    def _compute_V(self, i, n_index, n_quad, index_first, index_last):
        if self._conditional_linear_pack is None:
            self._conditional_linear_pack = _conditional_law_pack_linear_system(
                self._ijl2index,
                self._index2ijl,
                self._mark_probabilities,
                self._int_claw,
                self._IG,
                self._IG2,
            )
        index2ijl, ijl_to_index, _, int_x, int_y, int_sizes, *_ = self._conditional_linear_pack
        return _conditional_law_compute_V(
            i,
            n_index,
            n_quad,
            index_first,
            index_last,
            index2ijl,
            ijl_to_index,
            self._quad_x,
            int_x,
            int_y,
            int_sizes,
        )

    def _compute_M(self, n_index, n_quad, index_first, index_last, method):
        if self._conditional_linear_pack is None:
            self._conditional_linear_pack = _conditional_law_pack_linear_system(
                self._ijl2index,
                self._index2ijl,
                self._mark_probabilities,
                self._int_claw,
                self._IG,
                self._IG2,
            )
        (
            index2ijl,
            ijl_to_index,
            mark_probabilities,
            int_x,
            int_y,
            int_sizes,
            ig_x,
            ig_y,
            ig_sizes,
            ig2_x,
            ig2_y,
            ig2_sizes,
        ) = self._conditional_linear_pack
        return _conditional_law_compute_M(
            n_index,
            n_quad,
            index_first,
            index_last,
            method,
            index2ijl,
            ijl_to_index,
            self.mean_intensity,
            mark_probabilities,
            self._quad_x,
            self._quad_w,
            int_x,
            int_y,
            int_sizes,
            ig_x,
            ig_y,
            ig_sizes,
            ig2_x,
            ig2_y,
            ig2_sizes,
        )

    def _fill_M_for_gauss(self, M, method, n_quad, index_first, index, index1, j, l, j1, l1, fact, n, n1):
        def x_value(n_lower, n_greater, j_lower, j_greater, l_greater):
            return (
                self._mark_probabilities[j1][l1]
                * self._quad_w[n1]
                * self._G(j_lower, j_greater, l_greater, self._quad_x[n_greater] - self._quad_x[n_lower])
            )

        if n > n1:
            x = x_value(n1, n, j1, j, l)
        elif n < n1:
            x = fact * x_value(n, n1, j, j1, l1)
        else:
            if method == "gauss-":
                x = 0.0
            else:
                x1 = x_value(n1, n, j1, j, l)
                x2 = fact * x_value(n, n1, j, j1, l1)
                x = (x1 + x2) / 2.0
        if method == "gauss-":
            row = (index - index_first) * n_quad + n
            col = (index1 - index_first) * n_quad + n
            M[row, col] -= x
        if l == l1 and j == j1 and n == n1:
            x += 1.0
        row = (index - index_first) * n_quad + n
        col = (index1 - index_first) * n_quad + n1
        M[row, col] += x

    def _fill_M_for_log_lin(self, M, method, n_quad, index_first, index, index1, j, l, j1, l1, fact, n, n1):
        del method
        mark_probability = self._mark_probabilities[j1][l1]

        def ratio_dig(n_q):
            return (self._quad_x[n] - self._quad_x[n_q]) / self._quad_w[n_q]

        def ratio_dig2(n_q):
            return 1.0 / self._quad_w[n_q]

        def dig_arg_greater(n_q):
            return (
                j1,
                j,
                l,
                self._quad_x[n] - self._quad_x[n_q] - self._quad_w[n_q],
                self._quad_x[n] - self._quad_x[n_q],
            )

        def dig_arg_lower(n_q):
            return (
                j,
                j1,
                l1,
                self._quad_x[n_q] - self._quad_x[n],
                self._quad_x[n_q] - self._quad_x[n] + self._quad_w[n_q],
            )

        x = 0.0
        if n > n1:
            x += mark_probability * self._DIG(*dig_arg_greater(n1))
            if n1 < n_quad - 1:
                x -= ratio_dig(n1) * mark_probability * self._DIG(*dig_arg_greater(n1))
                x += ratio_dig2(n1) * mark_probability * self._DIG2(*dig_arg_greater(n1))
            if n1 > 0:
                x += ratio_dig(n1 - 1) * mark_probability * self._DIG(*dig_arg_greater(n1 - 1))
                x -= ratio_dig2(n1 - 1) * mark_probability * self._DIG2(*dig_arg_greater(n1 - 1))
        elif n < n1:
            x += fact * mark_probability * self._DIG(*dig_arg_lower(n1))
            if n1 < n_quad - 1:
                x -= fact * ratio_dig(n1) * mark_probability * self._DIG(*dig_arg_lower(n1))
                x -= fact * ratio_dig2(n1) * mark_probability * self._DIG2(*dig_arg_lower(n1))
            if n1 > 0:
                x += fact * ratio_dig(n1 - 1) * mark_probability * self._DIG(*dig_arg_lower(n1 - 1))
                x += fact * ratio_dig2(n1 - 1) * mark_probability * self._DIG2(*dig_arg_lower(n1 - 1))
        else:
            x += fact * self._mark_probabilities[j1][l1] * self._DIG(*dig_arg_lower(n1))
            if n1 < n_quad - 1:
                x -= fact * ratio_dig(n1) * mark_probability * self._DIG(*dig_arg_lower(n1))
                x -= fact * ratio_dig2(n1) * mark_probability * self._DIG2(*dig_arg_lower(n1))
            if n1 > 0:
                x += ratio_dig(n1 - 1) * mark_probability * self._DIG(*dig_arg_greater(n1 - 1))
                x -= ratio_dig2(n1 - 1) * mark_probability * self._DIG2(*dig_arg_greater(n1 - 1))
        if l == l1 and j == j1 and n == n1:
            x += 1.0
        row = (index - index_first) * n_quad + n
        col = (index1 - index_first) * n_quad + n1
        M[row, col] += x

    def _estimate_kernels_and_norms(self, i, index_first, index_last, res, n_quad, method):
        for index in range(index_first, index_last + 1):
            y = res[(index - index_first) * n_quad : (index - index_first + 1) * n_quad][:, 0]
            self._phi_ijl.append((self._quad_x, y))
            if method in {"gauss", "gauss-"}:
                self._norm_ijl.append(np.sum(y * self._quad_w))
            elif method in {"log", "lin"}:
                self._norm_ijl.append(np.sum((y[:-1] + y[1:]) / 2.0 * self._quad_w[:-1]))
        self.kernels.append([])
        for j in range(self.n_nodes):
            index = self._ijl2index[i][j][0]
            self.kernels[i].append(np.array(self._phi_ijl[index]) * self._mark_probabilities[j][0])
            self.kernels_norms[i, j] = self._norm_ijl[index] * self._mark_probabilities[j][0]
            index += 1
            for l in range(1, len(self.marked_components[j])):
                self.kernels[i][j] += np.array(self._phi_ijl[index]) * self._mark_probabilities[j][l]
                self.kernels_norms[i, j] += self._norm_ijl[index] * self._mark_probabilities[j][l]
                index += 1

    def _estimate_baseline(self):
        self.baseline = (np.eye(self.n_nodes) - self.kernels_norms).dot(self.mean_intensity)

    def _apply_final_kernel_symmetries(self):
        for group in self.symmetries2d:
            pairs = self._as_pair_group(group)
            if not pairs:
                continue
            avg_kernel = np.mean([self.kernels[i][j] for i, j in pairs], axis=0)
            avg_norm = float(np.mean([self.kernels_norms[i, j] for i, j in pairs]))
            for i, j in pairs:
                self.kernels[i][j] = avg_kernel.copy()
                self.kernels_norms[i, j] = avg_norm

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


class HawkesTheoreticalCumulant(BaseEstimator):
    """Theoretical Hawkes cumulants from Achab et al. equations 7-9."""

    def __init__(self, dim: int):
        self._dimension = int(dim)
        if self._dimension <= 0:
            raise ValueError("dim must be positive")
        self._baseline = None
        self._adjacency = None
        self._R_matrix = np.eye(self._dimension, dtype=float)
        self._mean_intensity = None
        self._covariance = None
        self._skewness = None

    @property
    def dimension(self):
        return self._dimension

    @property
    def baseline(self):
        return None if self._baseline is None else self._baseline.copy()

    @baseline.setter
    def baseline(self, mu):
        arr = np.asarray(mu, dtype=float)
        if arr.shape != (self.dimension,):
            raise ValueError(f"baseline must have shape ({self.dimension},)")
        self._baseline = arr.copy()
        self._mean_intensity = None
        self._covariance = None
        self._skewness = None

    @property
    def adjacency(self):
        return None if self._adjacency is None else self._adjacency.copy()

    @adjacency.setter
    def adjacency(self, adjacency):
        arr = np.asarray(adjacency, dtype=float)
        if arr.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"adjacency must have shape ({self.dimension}, {self.dimension})"
            )
        self._adjacency = arr.copy()
        self._R_matrix = scipy.linalg.inv(np.eye(self.dimension, dtype=float) - arr)
        self._mean_intensity = None
        self._covariance = None
        self._skewness = None

    @property
    def _R(self):
        return self._R_matrix.copy()

    @property
    def mean_intensity(self):
        if self._mean_intensity is None:
            self.compute_mean_intensity()
        return self._mean_intensity.copy()

    @property
    def covariance(self):
        if self._covariance is None:
            self.compute_covariance()
        return self._covariance.copy()

    @property
    def skewness(self):
        if self._skewness is None:
            self.compute_skewness()
        return self._skewness.copy()

    def _check_ready(self):
        if self._baseline is None:
            raise ValueError("baseline must be set before computing cumulants")

    def compute_mean_intensity(self):
        self._check_ready()
        self._mean_intensity = self._R_matrix @ self._baseline

    def compute_covariance(self):
        if self._mean_intensity is None:
            self.compute_mean_intensity()
        self._covariance = self._R_matrix @ np.diag(self._mean_intensity) @ self._R_matrix.T

    def compute_skewness(self):
        if self._covariance is None:
            self.compute_covariance()
        d = self.dimension
        out = np.zeros((d, d), dtype=float)
        for i in range(d):
            for k in range(d):
                value = 0.0
                for m in range(d):
                    r_im = self._R_matrix[i, m]
                    r_km = self._R_matrix[k, m]
                    value += (
                        r_im * r_im * self._covariance[k, m]
                        + 2.0 * r_im * r_km * self._covariance[i, m]
                        - 2.0 * self._mean_intensity[m] * r_im * r_im * r_km
                    )
                out[i, k] = value
        self._skewness = out

    def compute_cumulants(self):
        self.compute_mean_intensity()
        self.compute_covariance()
        self.compute_skewness()


def _cumulant_compute_A_and_I_reference(events, sizes, i, j, end_time, support, mean_intensity_j):
    n_j = int(sizes[j])
    res_C = 0.0
    res_J = 0.0
    width = 2.0 * float(support)
    trend_C_j = float(mean_intensity_j) * width
    trend_J_j = float(mean_intensity_j) * width * width
    last_l = 0
    for event_index in range(int(sizes[i])):
        t_i_k = float(events[i, event_index])
        if t_i_k - support < 0:
            continue
        while last_l < n_j and float(events[j, last_l]) <= t_i_k - width:
            last_l += 1
        l = last_l
        timestamps_in_interval = 0
        sub_res = 0.0
        while l < n_j:
            abs_delta = abs(float(events[j, l]) - t_i_k)
            if abs_delta < width:
                sub_res += width - abs_delta
                if abs_delta < support:
                    timestamps_in_interval += 1
            else:
                break
            l += 1
        if l == n_j:
            continue
        res_C += timestamps_in_interval - trend_C_j
        res_J += sub_res - trend_J_j
    return res_C / float(end_time), res_J / float(end_time)


def _cumulant_compute_E_reference(events, sizes, i, j, k, end_time, support, mean_intensity_i, mean_intensity_j, J_ij):
    n_i = int(sizes[i])
    n_j = int(sizes[j])
    res = 0.0
    last_l = 0
    last_m = 0
    trend_i = float(mean_intensity_i) * 2.0 * float(support)
    trend_j = float(mean_intensity_j) * 2.0 * float(support)
    for tau_index in range(int(sizes[k])):
        tau = float(events[k, tau_index])
        if tau - support < 0:
            continue
        while last_l < n_i and float(events[i, last_l]) <= tau - support:
            last_l += 1
        l = last_l
        while l < n_i and float(events[i, l]) < tau + support:
            l += 1
        while last_m < n_j and float(events[j, last_m]) <= tau - support:
            last_m += 1
        m = last_m
        while m < n_j and float(events[j, m]) < tau + support:
            m += 1
        if m == n_j or l == n_i:
            continue
        res += (l - last_l - trend_i) * (m - last_m - trend_j) - float(J_ij)
    return res / float(end_time)


def _cumulant_compute_A_and_I_numba_impl(events, sizes, i, j, end_time, support, mean_intensity_j):
    n_j = sizes[j]
    res_C = 0.0
    res_J = 0.0
    width = 2.0 * support
    trend_C_j = mean_intensity_j * width
    trend_J_j = mean_intensity_j * width * width
    last_l = 0
    for event_index in range(sizes[i]):
        t_i_k = events[i, event_index]
        if t_i_k - support < 0.0:
            continue
        while last_l < n_j and events[j, last_l] <= t_i_k - width:
            last_l += 1
        l = last_l
        timestamps_in_interval = 0
        sub_res = 0.0
        while l < n_j:
            abs_delta = abs(events[j, l] - t_i_k)
            if abs_delta < width:
                sub_res += width - abs_delta
                if abs_delta < support:
                    timestamps_in_interval += 1
            else:
                break
            l += 1
        if l == n_j:
            continue
        res_C += timestamps_in_interval - trend_C_j
        res_J += sub_res - trend_J_j
    return res_C / end_time, res_J / end_time


def _cumulant_compute_E_numba_impl(events, sizes, i, j, k, end_time, support, mean_intensity_i, mean_intensity_j, J_ij):
    n_i = sizes[i]
    n_j = sizes[j]
    res = 0.0
    last_l = 0
    last_m = 0
    trend_i = mean_intensity_i * 2.0 * support
    trend_j = mean_intensity_j * 2.0 * support
    for tau_index in range(sizes[k]):
        tau = events[k, tau_index]
        if tau - support < 0.0:
            continue
        while last_l < n_i and events[i, last_l] <= tau - support:
            last_l += 1
        l = last_l
        while l < n_i and events[i, l] < tau + support:
            l += 1
        while last_m < n_j and events[j, last_m] <= tau - support:
            last_m += 1
        m = last_m
        while m < n_j and events[j, m] < tau + support:
            m += 1
        if m == n_j or l == n_i:
            continue
        res += (l - last_l - trend_i) * (m - last_m - trend_j) - J_ij
    return res / end_time


_cumulant_compute_A_and_I_numba = _compile(_cumulant_compute_A_and_I_numba_impl)
_cumulant_compute_E_numba = _compile(_cumulant_compute_E_numba_impl)


def _cumulant_compute_A_and_I(events, sizes, i, j, end_time, support, mean_intensity_j):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    i = int(i)
    j = int(j)
    if NUMBA_AVAILABLE:
        return _cumulant_compute_A_and_I_numba(
            events,
            sizes,
            i,
            j,
            float(end_time),
            float(support),
            float(mean_intensity_j),
        )
    return _cumulant_compute_A_and_I_reference(events, sizes, i, j, end_time, support, mean_intensity_j)


def _cumulant_compute_E(events, sizes, i, j, k, end_time, support, mean_intensity_i, mean_intensity_j, J_ij):
    events = np.ascontiguousarray(np.asarray(events, dtype=float))
    sizes = np.ascontiguousarray(np.asarray(sizes, dtype=np.int64))
    i = int(i)
    j = int(j)
    k = int(k)
    if NUMBA_AVAILABLE:
        return _cumulant_compute_E_numba(
            events,
            sizes,
            i,
            j,
            k,
            float(end_time),
            float(support),
            float(mean_intensity_i),
            float(mean_intensity_j),
            float(J_ij),
        )
    return _cumulant_compute_E_reference(
        events,
        sizes,
        i,
        j,
        k,
        end_time,
        support,
        mean_intensity_i,
        mean_intensity_j,
        J_ij,
    )


class _HawkesCumulantComputer:
    """Pure-Python port of tick's Hawkes cumulant C++ helper."""

    def __init__(self, integration_support=100.0):
        self._integration_support = None
        self.integration_support = integration_support
        self.L = None
        self.C = None
        self.K_c = None
        self._L_day = None
        self._J = None
        self._events_of_cumulants = None
        self._are_cumulants_ready = False
        self._realizations = []
        self._end_times = np.empty(0, dtype=float)
        self._n_nodes = 0
        self._packed_realizations = []
        self._packed_sizes = []

    @property
    def integration_support(self):
        return self._integration_support

    @integration_support.setter
    def integration_support(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("Kernel support must be positive")
        if self._integration_support != value:
            self._are_cumulants_ready = False
        self._integration_support = value

    def set_data(self, realizations, end_times, n_nodes=None):
        self._realizations = [] if realizations is None else realizations
        self._end_times = (
            np.empty(0, dtype=float)
            if end_times is None
            else np.asarray(end_times, dtype=float)
        )
        if n_nodes is None:
            n_nodes = len(self._realizations[0]) if self._realizations else 0
        self._n_nodes = int(n_nodes)
        self._packed_realizations = []
        self._packed_sizes = []

    @staticmethod
    def _same_realizations(events_1, events_2):
        if len(events_1) != len(events_2):
            return False
        for r, realization_1 in enumerate(events_1):
            realization_2 = events_2[r]
            if len(realization_1) != len(realization_2):
                return False
            for i, timestamps_1 in enumerate(realization_1):
                if timestamps_1.__array_interface__ != realization_2[i].__array_interface__:
                    return False
        return True

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_realizations(self):
        return len(self._realizations)

    @property
    def realizations(self):
        return self._realizations

    @property
    def end_times(self):
        return self._end_times

    @property
    def cumulants_ready(self):
        events_didnt_change = (
            self._events_of_cumulants is not None
            and _HawkesCumulantComputer._same_realizations(
                self._events_of_cumulants, self.realizations
            )
        )
        return self._are_cumulants_ready and events_didnt_change

    def compute_cumulants(self, verbose=False, force=False):
        if len(self.realizations) == 0:
            raise RuntimeError("Cannot compute cumulants if no realization has been provided")
        if self.cumulants_ready and not force:
            if verbose:
                print("Use previouly computed cumulants")
            return
        self._events_of_cumulants = self.realizations
        self._pack_realizations()
        self.compute_L()
        self.compute_C_and_J()
        self.K_c = self.compute_E_c()
        self._are_cumulants_ready = True

    def _pack_realizations(self):
        self._packed_realizations = []
        self._packed_sizes = []
        for realization in self.realizations:
            events, sizes = pack_realization(realization)
            self._packed_realizations.append(events)
            self._packed_sizes.append(sizes)

    def compute_L(self):
        self._L_day = np.zeros((self.n_realizations, self.n_nodes), dtype=float)
        for day, realization in enumerate(self.realizations):
            for i in range(self.n_nodes):
                self._L_day[day, i] = len(realization[i]) / self.end_times[day]
        self.L = np.mean(self._L_day, axis=0)

    def compute_C_and_J(self):
        if len(self._packed_realizations) != self.n_realizations:
            self._pack_realizations()
        d = self.n_nodes
        self.C = np.zeros((d, d), dtype=float)
        self._J = np.zeros((self.n_realizations, d, d), dtype=float)
        for day in range(self.n_realizations):
            events = self._packed_realizations[day]
            sizes = self._packed_sizes[day]
            C_day = np.zeros((d, d), dtype=float)
            J_day = np.zeros((d, d), dtype=float)
            for i, j in product(range(d), repeat=2):
                C_day[i, j], J_day[i, j] = _cumulant_compute_A_and_I(
                    events,
                    sizes,
                    i,
                    j,
                    self.end_times[day],
                    self.integration_support,
                    self._L_day[day, j],
                )
            C_day[:] = 0.5 * (C_day + C_day.T)
            J_day[:] = 0.5 * (J_day + J_day.T)
            self.C += C_day / self.n_realizations
            self._J[day] = J_day

    def compute_E_c(self):
        if len(self._packed_realizations) != self.n_realizations:
            self._pack_realizations()
        d = self.n_nodes
        E_c = np.zeros((d, d, 2), dtype=float)
        for day in range(self.n_realizations):
            events = self._packed_realizations[day]
            sizes = self._packed_sizes[day]
            for i in range(d):
                for j in range(d):
                    E_c[i, j, 0] += _cumulant_compute_E(
                        events,
                        sizes,
                        i,
                        j,
                        j,
                        self.end_times[day],
                        self.integration_support,
                        self._L_day[day, i],
                        self._L_day[day, j],
                        self._J[day, i, j],
                    )
                    E_c[i, j, 1] += _cumulant_compute_E(
                        events,
                        sizes,
                        j,
                        j,
                        i,
                        self.end_times[day],
                        self.integration_support,
                        self._L_day[day, j],
                        self._L_day[day, j],
                        self._J[day, j, j],
                    )
        E_c /= self.n_realizations
        return (2.0 * E_c[:, :, 0] + E_c[:, :, 1]) / 3.0

    @staticmethod
    def _compute_A_and_I_ij(timestamps_i, timestamps_j, end_time, support, mean_intensity_j):
        events, sizes = pack_realization([timestamps_i, timestamps_j])
        return _cumulant_compute_A_and_I(events, sizes, 0, 1, end_time, support, mean_intensity_j)

    @staticmethod
    def _compute_E_ijk(
        timestamps_i,
        timestamps_j,
        timestamps_k,
        end_time,
        support,
        mean_intensity_i,
        mean_intensity_j,
        J_ij,
    ):
        events, sizes = pack_realization([timestamps_i, timestamps_j, timestamps_k])
        return _cumulant_compute_E(
            events,
            sizes,
            0,
            1,
            2,
            end_time,
            support,
            mean_intensity_i,
            mean_intensity_j,
            J_ij,
        )


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
        self._cumulant_computer = _HawkesCumulantComputer(integration_support)

    def _set_data(self, events, end_times=None):
        super()._set_data(events, end_times)
        self._cumulant_computer.set_data(self.data, self._end_times, self._n_nodes)

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
        self._cumulant_computer.integration_support = self.integration_support
        self._cumulant_computer.compute_cumulants(verbose=self.verbose, force=force)
        self._mean_intensity = self._cumulant_computer.L
        self._covariance = self._cumulant_computer.C
        self._skewness = self._cumulant_computer.K_c

    @property
    def mean_intensity(self):
        if not self._cumulant_computer.cumulants_ready:
            self.compute_cumulants()
        return self._mean_intensity.copy()

    @property
    def covariance(self):
        if not self._cumulant_computer.cumulants_ready:
            self.compute_cumulants()
        return self._covariance.copy()

    @property
    def skewness(self):
        if not self._cumulant_computer.cumulants_ready:
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
