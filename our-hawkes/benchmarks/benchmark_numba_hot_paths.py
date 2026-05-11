"""Smoke benchmark for optional Numba hot paths.

This script is intentionally lightweight: it separates first-call wrapper time,
warm wrapper time, and direct reference time, but it does not enforce speed
thresholds. It can be run from a checkout without installing the package.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from our_hawkes.hawkes import (  # noqa: E402
    HawkesADM4,
    HawkesBasisKernels,
    HawkesEM,
    HawkesSumGaussians,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLogLik,
    SimuHawkesExpKernels,
    SimuHawkesSumExpKernels,
    SimuPoissonProcess,
)
from our_hawkes.hawkes import numeric  # noqa: E402


def _time_once(func):
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def _time_average(func, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return (time.perf_counter() - start) / repeats


def _print_timing(name, cold, warm, reference):
    print(f"{name}: cold={cold:.8f}s warm={warm:.8f}s reference={reference:.8f}s")


def _fixture():
    realization = [
        np.array([0.1, 0.7, 1.4, 2.0]),
        np.array([0.2, 0.9, 1.6, 2.3]),
    ]
    events, sizes = numeric.pack_realization(realization)
    end_time = 2.5
    baseline = np.array([0.35, 0.45])
    exp_adjacency = np.array([[0.12, 0.05], [0.08, 0.16]])
    exp_decays = np.array([[1.1, 1.7], [0.9, 1.3]])
    sumexp_adjacency = np.array(
        [
            [[0.08, 0.03], [0.04, 0.02]],
            [[0.05, 0.01], [0.07, 0.06]],
        ]
    )
    sumexp_decays = np.array([0.7, 1.4])
    return realization, events, sizes, end_time, baseline, exp_adjacency, exp_decays, sumexp_adjacency, sumexp_decays


def benchmark_numeric():
    realization, events, sizes, end_time, baseline, exp_adj, exp_decays, sum_adj, sum_decays = _fixture()
    del realization
    poisson_uniforms = np.random.default_rng(123).random((64, 2))
    poisson_intensities = np.array([0.4, 0.8])

    def poisson_wrapper():
        out_times = np.empty(poisson_uniforms.shape[0], dtype=float)
        out_nodes = np.empty(poisson_uniforms.shape[0], dtype=np.int64)
        return numeric.homogeneous_poisson_events(
            0.0, 20.0, poisson_intensities, poisson_uniforms, out_times, out_nodes
        )

    def poisson_reference():
        out_times = np.empty(poisson_uniforms.shape[0], dtype=float)
        out_nodes = np.empty(poisson_uniforms.shape[0], dtype=np.int64)
        return numeric.homogeneous_poisson_events_reference(
            0.0, 20.0, poisson_intensities, poisson_uniforms, out_times, out_nodes
        )

    targets = [
        (
            "numeric.homogeneous_poisson_events",
            poisson_wrapper,
            poisson_reference,
        ),
        (
            "numeric.exp_intensity_bound",
            lambda: numeric.exp_intensity_bound(1.4, events, sizes, baseline, exp_adj, exp_decays),
            lambda: numeric.exp_intensity_bound_reference(1.4, events, sizes, baseline, exp_adj, exp_decays),
        ),
        (
            "numeric.sumexp_intensity_bound",
            lambda: numeric.sumexp_intensity_bound(1.4, events, sizes, baseline, sum_adj, sum_decays),
            lambda: numeric.sumexp_intensity_bound_reference(1.4, events, sizes, baseline, sum_adj, sum_decays),
        ),
        (
            "numeric.exp_loglik_loss_scan",
            lambda: numeric.exp_loglik_loss_scan(events, sizes, end_time, baseline, exp_adj, exp_decays),
            lambda: numeric.exp_loglik_loss_scan_reference(events, sizes, end_time, baseline, exp_adj, exp_decays),
        ),
        (
            "numeric.sumexp_loglik_loss_scan",
            lambda: numeric.sumexp_loglik_loss_scan(events, sizes, end_time, baseline, sum_adj, sum_decays),
            lambda: numeric.sumexp_loglik_loss_scan_reference(events, sizes, end_time, baseline, sum_adj, sum_decays),
        ),
        (
            "numeric.exp_loglik_grad_scan",
            lambda: numeric.exp_loglik_grad_scan(events, sizes, end_time, baseline, exp_adj, exp_decays),
            lambda: numeric.exp_loglik_grad_scan_reference(events, sizes, end_time, baseline, exp_adj, exp_decays),
        ),
        (
            "numeric.sumexp_loglik_grad_scan",
            lambda: numeric.sumexp_loglik_grad_scan(events, sizes, end_time, baseline, sum_adj, sum_decays),
            lambda: numeric.sumexp_loglik_grad_scan_reference(events, sizes, end_time, baseline, sum_adj, sum_decays),
        ),
        (
            "numeric.exp_ls_statistics",
            lambda: numeric.exp_ls_statistics(events, sizes, end_time, exp_decays[0], 0),
            lambda: numeric.exp_ls_statistics_reference(events, sizes, end_time, exp_decays[0], 0),
        ),
    ]
    for name, wrapper, reference in targets:
        _print_timing(name, _time_once(wrapper), _time_average(wrapper), _time_average(reference))


def benchmark_models_and_simulation():
    realization, events, sizes, end_time, baseline, exp_adj, exp_decays, sum_adj, sum_decays = _fixture()
    del events, sizes
    exp_model = ModelHawkesExpKernLogLik(1.3).fit(realization, end_time)
    exp_coeffs = np.hstack((baseline, exp_adj.ravel()))
    sum_model = ModelHawkesSumExpKernLogLik(sum_decays).fit(realization, end_time)
    sum_coeffs = np.hstack((baseline, sum_adj.ravel()))

    targets = [
        ("model.exp.loss", lambda: exp_model.loss(exp_coeffs), lambda: exp_model.loss(exp_coeffs)),
        ("model.exp.grad", lambda: exp_model.grad(exp_coeffs), lambda: exp_model.grad(exp_coeffs)),
        ("model.sumexp.loss", lambda: sum_model.loss(sum_coeffs), lambda: sum_model.loss(sum_coeffs)),
        ("model.sumexp.grad", lambda: sum_model.grad(sum_coeffs), lambda: sum_model.grad(sum_coeffs)),
    ]
    for name, wrapper, reference in targets:
        _print_timing(name, _time_once(wrapper), _time_average(wrapper), _time_average(reference))

    exp_simu = SimuHawkesExpKernels(exp_adj, exp_decays, baseline=baseline, verbose=False).set_timestamps(
        realization, end_time
    )
    sum_simu = SimuHawkesSumExpKernels(sum_adj, sum_decays, baseline=baseline, verbose=False).set_timestamps(
        realization, end_time
    )
    sim_targets = [
        ("simulation.poisson.simulate", lambda: SimuPoissonProcess([0.4, 0.8], end_time=20.0, seed=7, verbose=False).simulate(), lambda: SimuPoissonProcess([0.4, 0.8], end_time=20.0, seed=7, verbose=False).simulate()),
        ("simulation.exp.intensity", lambda: exp_simu._intensity_at(1.4), lambda: exp_simu._intensity_at(1.4)),
        ("simulation.exp.bound", lambda: exp_simu._total_intensity_bound(1.4), lambda: numeric.exp_intensity_bound_reference(1.4, *numeric.pack_realization(exp_simu.timestamps), baseline, exp_adj, exp_decays)),
        ("simulation.exp.compensator", lambda: exp_simu._evaluate_compensator(0, end_time), lambda: exp_simu._evaluate_compensator(0, end_time)),
        ("simulation.sumexp.intensity", lambda: sum_simu._intensity_at(1.4), lambda: sum_simu._intensity_at(1.4)),
        ("simulation.sumexp.bound", lambda: sum_simu._total_intensity_bound(1.4), lambda: numeric.sumexp_intensity_bound_reference(1.4, *numeric.pack_realization(sum_simu.timestamps), baseline, sum_adj, sum_decays)),
        ("simulation.sumexp.compensator", lambda: sum_simu._evaluate_compensator(0, end_time), lambda: sum_simu._evaluate_compensator(0, end_time)),
    ]
    for name, wrapper, reference in sim_targets:
        _print_timing(name, _time_once(wrapper), _time_average(wrapper), _time_average(reference))


def benchmark_learners():
    realization, _, _, end_time, baseline, exp_adj, exp_decays, _, _ = _fixture()
    del baseline, exp_adj, exp_decays
    learners = [
        ("learner.HawkesEM.fit", lambda: HawkesEM(1.0, kernel_size=4, max_iter=2).fit(realization, end_time)),
        ("learner.HawkesADM4.fit", lambda: HawkesADM4(1.0, max_iter=2, C=10.0, verbose=False).fit(realization, end_time)),
        ("learner.HawkesSumGaussians.fit", lambda: HawkesSumGaussians(1.0, n_gaussians=3, max_iter=2).fit(realization, end_time)),
        ("learner.HawkesBasisKernels.fit", lambda: HawkesBasisKernels(1.0, kernel_size=4, max_iter=2).fit(realization, end_time)),
    ]
    for name, wrapper in learners:
        _print_timing(name, _time_once(wrapper), _time_average(wrapper, repeats=2), _time_average(wrapper, repeats=2))


def main():
    print(f"numba_enabled={numeric.NUMBA_AVAILABLE}")
    benchmark_numeric()
    benchmark_models_and_simulation()
    benchmark_learners()


if __name__ == "__main__":
    main()
