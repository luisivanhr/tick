import math
import sys
import unittest
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes import plot as hawkes_plot
from our_hawkes.base import TimeFunction
from our_hawkes.hawkes import (
    HawkesCumulantMatching,
    HawkesEM,
    HawkesExpKern,
    HawkesKernel0,
    HawkesKernelExp,
    HawkesKernelPowerLaw,
    HawkesKernelSumExp,
    HawkesKernelTimeFunc,
    HawkesSumExpKern,
    ModelHawkesSumExpKernLeastSq,
    SimuHawkes,
    SimuHawkesExpKernels,
    SimuHawkesMulti,
    SimuHawkesSumExpKernels,
    SimuInhomogeneousPoisson,
    SimuPoissonProcess,
)


class TickTimeFunctionCompatibilityTest(unittest.TestCase):
    def test_time_function_constants_interpolation_and_borders(self):
        constant = TimeFunction(2.5)
        self.assertEqual(constant.value(-1.0), 0.0)
        self.assertEqual(constant.value(3.0), 2.5)
        self.assertTrue(math.isinf(constant.get_norm()))

        t_values = np.array([0.0, 1.0, 2.0])
        y_values = np.array([0.0, 2.0, 4.0])
        linear = TimeFunction((t_values, y_values))
        self.assertAlmostEqual(linear.dt, 0.2)
        self.assertAlmostEqual(linear.value(0.5), 1.0)
        self.assertEqual(linear.value(3.0), 0.0)

        const_left = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstLeft)
        const_right = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight)
        self.assertEqual(const_left.value(0.5), 2.0)
        self.assertEqual(const_right.value(0.5), 0.0)

        border_constant = TimeFunction(
            (t_values, y_values),
            border_type=TimeFunction.BorderConstant,
            border_value=7.0,
        )
        border_continue = TimeFunction((t_values, y_values), border_type=TimeFunction.BorderContinue)
        cyclic = TimeFunction((t_values, y_values), border_type=TimeFunction.Cyclic)
        self.assertEqual(border_constant.value(3.0), 7.0)
        self.assertEqual(border_continue.value(3.0), 4.0)
        self.assertAlmostEqual(cyclic.value(2.5), linear.value(0.5))


class TickPointProcessCompatibilityTest(unittest.TestCase):
    def test_poisson_max_jumps_and_intensity_tracking_toggle(self):
        process = SimuPoissonProcess([1.0, 3.0, 2.2], end_time=None, max_jumps=25, seed=123, verbose=False)
        self.assertEqual(process.simulation_time, 0.0)
        self.assertEqual(process.n_total_jumps, 0)
        self.assertFalse(process.is_intensity_tracked())

        process.track_intensity(0.1)
        self.assertTrue(process.is_intensity_tracked())
        process.track_intensity(-0.1)
        self.assertFalse(process.is_intensity_tracked())

        process.simulate()
        self.assertEqual(process.n_total_jumps, 25)
        self.assertGreater(process.simulation_time, 0.0)
        self.assertIsNone(process.end_time)
        self.assertEqual(sum(map(len, process.timestamps)), process.n_total_jumps)

    def test_inhomogeneous_poisson_tracking_matches_time_functions(self):
        t_values = np.linspace(0.0, 5.0, 12)
        tf_1 = TimeFunction((t_values, np.maximum(0.5 + np.sin(t_values), 0.0)))
        tf_2 = TimeFunction((t_values, 1.0 / (1.0 + t_values)))

        process = SimuInhomogeneousPoisson([tf_1, tf_2], end_time=5.0, seed=2937, verbose=False)
        process.track_intensity(0.2)
        process.simulate()

        times = process.intensity_tracked_times
        np.testing.assert_allclose(process.tracked_intensity[0], tf_1.value(times), atol=1e-12)
        np.testing.assert_allclose(process.tracked_intensity[1], tf_2.value(times), atol=1e-12)


class TickGenericHawkesCompatibilityTest(unittest.TestCase):
    def setUp(self):
        t_values = np.linspace(0.0, 10.0, 10)
        y_values = np.maximum(0.5 + np.sin(t_values), 0.0)
        self.time_func_kernel = HawkesKernelTimeFunc(t_values=t_values, y_values=y_values)
        self.kernels = np.array(
            [
                [HawkesKernel0(), HawkesKernelExp(0.1, 3.0)],
                [HawkesKernelPowerLaw(0.2, 4.0, 2.0), HawkesKernelSumExp([0.1, 0.4], [3.0, 4.0])],
            ],
            dtype=object,
        )
        self.baseline = np.array([0.4, 0.7])

    def test_generic_hawkes_constructor_and_setters(self):
        hawkes = SimuHawkes(kernels=self.kernels, verbose=False)
        self.assertEqual(hawkes.n_nodes, 2)
        np.testing.assert_array_equal(hawkes.baseline, np.zeros(2))
        for i, j in product(range(2), range(2)):
            self.assertIs(hawkes.kernels[i, j], self.kernels[i, j])

        hawkes = SimuHawkes(baseline=self.baseline, verbose=False)
        self.assertEqual(hawkes.n_nodes, 2)
        np.testing.assert_array_equal(hawkes.baseline, self.baseline)
        first_zero = hawkes.kernels[0, 0]
        for i, j in product(range(2), range(2)):
            self.assertIsInstance(hawkes.kernels[i, j], HawkesKernel0)
            self.assertIs(hawkes.kernels[i, j], first_zero)

        hawkes = SimuHawkes(n_nodes=2, verbose=False)
        for i, j in product(range(2), range(2)):
            hawkes.set_kernel(i, j, self.kernels[i, j])
            self.assertIs(hawkes.kernels[i, j], self.kernels[i, j])
        hawkes.set_kernel(1, 1, self.time_func_kernel)
        self.assertIs(hawkes.kernels[1, 1], self.time_func_kernel)
        hawkes.set_baseline(0, 1.25)
        self.assertEqual(hawkes.baseline[0], 1.25)

    def test_generic_hawkes_piecewise_and_timefunction_baselines(self):
        baselines = [[1.0, 2.0, 1.5, 4.0], [2.0, 1.5, 4.0, 1.0]]
        hawkes = SimuHawkes(baseline=baselines, period_length=4.0, kernels=self.kernels, verbose=False)
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.9, 4.1])
        np.testing.assert_allclose(
            hawkes.get_baseline_values(0, times),
            np.array([1.0, 1.0, 2.0, 2.0, 1.5, 4.0, 1.0]),
        )

        tf_1 = TimeFunction(([0.0, 1.0, 2.0], [0.0, 2.0, 0.0]))
        tf_2 = TimeFunction(([0.0, 1.0, 2.0], [1.0, 0.0, 1.0]))
        hawkes = SimuHawkes(baseline=[tf_1, tf_2], kernels=self.kernels, verbose=False)
        np.testing.assert_allclose(hawkes.get_baseline_values(0, times), tf_1.value(times))

    def test_sumexp_simulation_kernel_matrix_radius_and_mean(self):
        np.random.seed(23982)
        n_nodes = 3
        n_decays = 4
        baseline = np.random.rand(n_nodes)
        adjacency = np.random.rand(n_nodes, n_nodes, n_decays) / 10.0
        decays = np.random.rand(n_decays)
        adjacency[0, 0, :] = 0.0
        adjacency[-1, -1, :] = 0.0

        hawkes = SimuHawkesSumExpKernels(adjacency, decays, baseline=baseline, seed=203, verbose=False)
        zero_kernel = None
        for i, j in product(range(n_nodes), range(n_nodes)):
            kernel = hawkes.kernels[i, j]
            if np.linalg.norm(adjacency[i, j, :]) == 0:
                self.assertIsInstance(kernel, HawkesKernel0)
                zero_kernel = kernel if zero_kernel is None else zero_kernel
                self.assertIs(kernel, zero_kernel)
            else:
                self.assertIsInstance(kernel, HawkesKernelSumExp)
                np.testing.assert_array_equal(kernel.decays, decays)
                np.testing.assert_array_equal(kernel.intensities, adjacency[i, j, :])

        expected_radius = max(abs(np.linalg.eigvals(adjacency.sum(axis=2))))
        self.assertAlmostEqual(hawkes.spectral_radius(), expected_radius)
        hawkes.adjust_spectral_radius(0.6)
        self.assertAlmostEqual(hawkes.spectral_radius(), 0.6)
        np.testing.assert_allclose(
            hawkes.mean_intensity(),
            np.linalg.solve(np.eye(n_nodes) - hawkes.adjacency.sum(axis=2), baseline),
        )

    def test_set_timestamps_then_simulate_resumes_from_manual_history(self):
        hawkes = SimuHawkesExpKernels([[0.2, 0.1], [0.0, 0.3]], 0.5, baseline=[0.2, 0.4], verbose=False)
        original_timestamps = [np.array([7.096244, 9.389927]), np.array([0.436199, 0.659153])]
        hawkes.set_timestamps(original_timestamps, 10.0)
        hawkes.end_time = 20.0
        hawkes.simulate()
        for actual, expected in zip(hawkes.timestamps, original_timestamps):
            np.testing.assert_allclose(actual[: expected.size], expected)


class TickModelPublicBehaviorTest(unittest.TestCase):
    def test_sumexp_leastsq_parameter_validation_and_baseline_intervals(self):
        decays = np.array([1.0, 2.0])
        self.assertEqual(ModelHawkesSumExpKernLeastSq(decays).n_decays, 2)

        with self.assertRaisesRegex(ValueError, "n_baselines must be positive"):
            ModelHawkesSumExpKernLeastSq(decays, n_baselines=-1, period_length=2.0)
        with self.assertRaisesRegex(ValueError, "period_length must be given"):
            ModelHawkesSumExpKernLeastSq(decays, n_baselines=3)
        with self.assertWarnsRegex(UserWarning, "period_length has no effect"):
            ModelHawkesSumExpKernLeastSq(decays, period_length=2.0)

        model = ModelHawkesSumExpKernLeastSq(decays, n_baselines=4, period_length=10.0)
        np.testing.assert_array_equal(model.baseline_intervals, np.array([0.0, 2.5, 5.0, 7.5]))
        np.testing.assert_array_equal(ModelHawkesSumExpKernLeastSq(decays).baseline_intervals, np.array([0.0]))

    def test_incremental_fit_matches_list_fit_for_sumexp_leastsq(self):
        events = [
            [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])],
            [np.array([0.1, 0.6]), np.array([0.4, 1.4, 1.9])],
        ]
        end_times = np.array([2.0, 2.2])
        coeffs = np.array([0.3, 0.4, 0.1, 0.05, 0.02, 0.03, 0.04, 0.01, 0.05, 0.02])

        batch = ModelHawkesSumExpKernLeastSq([1.0, 2.0]).fit(events, end_times)
        incremental = ModelHawkesSumExpKernLeastSq([1.0, 2.0])
        incremental.incremental_fit(events[0], end_times[0])
        incremental.incremental_fit(events[1], end_times[1])

        self.assertEqual(batch.n_nodes, incremental.n_nodes)
        np.testing.assert_allclose(batch.end_times, incremental.end_times)
        self.assertAlmostEqual(batch.loss(coeffs), incremental.loss(coeffs), places=12)
        np.testing.assert_allclose(batch.grad(coeffs), incremental.grad(coeffs), atol=1e-12)

    def test_sumexp_leastsq_cast_period_length(self):
        model = ModelHawkesSumExpKernLeastSq([1.0, 2.0], n_baselines=2, period_length=10.0)
        self.assertEqual(model.cast_period_length(5.0), 5.0)


class TickLearnerPublicBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.events = [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])]
        self.end_time = 2.0

    def test_exp_and_sumexp_learners_score_and_kernel_accessors(self):
        exp = HawkesExpKern(1.2, gofit="likelihood", penalty="none", max_iter=3, verbose=False)
        exp.fit(self.events, end_times=self.end_time)
        self.assertEqual(exp.baseline.shape, (2,))
        self.assertEqual(exp.adjacency.shape, (2, 2))
        self.assertTrue(np.isfinite(exp.score()))
        np.testing.assert_allclose(exp.get_kernel_norms(), exp.adjacency, atol=1e-12)
        self.assertEqual(exp.get_kernel_values(0, 1, np.linspace(0.0, 1.0, 5)).shape, (5,))

        sumexp = HawkesSumExpKern([1.0, 2.0], penalty="none", max_iter=3, verbose=False)
        sumexp.fit(self.events, end_times=self.end_time)
        self.assertEqual(sumexp.baseline.shape, (2,))
        self.assertEqual(sumexp.adjacency.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(sumexp.score()))
        np.testing.assert_allclose(sumexp.get_kernel_norms(), sumexp.adjacency.sum(axis=2), atol=1e-12)
        self.assertEqual(sumexp.get_baseline_values(0, np.linspace(0.0, 2.0, 4)).shape, (4,))

    def test_em_time_changed_interarrival_times_and_score(self):
        em = HawkesEM(kernel_support=1.0, kernel_size=4, max_iter=3, verbose=False)
        em.fit(self.events, end_times=self.end_time)
        self.assertEqual(em.n_nodes, 2)
        self.assertEqual(em.n_realizations, 1)
        self.assertTrue(np.isfinite(em.score()))
        residuals = em.time_changed_interarrival_times()
        self.assertEqual(len(residuals), 1)
        self.assertEqual(len(residuals[0]), 2)
        for node_residuals in residuals[0]:
            self.assertTrue(np.all(np.isfinite(node_residuals)))
            self.assertTrue(np.all(node_residuals >= 0.0))

    def test_cumulant_matching_public_diagnostics(self):
        events = [
            [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])],
            [np.array([0.1, 0.6]), np.array([0.4, 1.4, 1.9])],
        ]
        learner = HawkesCumulantMatching(1.0).fit(events, end_times=[2.0, 2.2])
        self.assertEqual(learner.mean_intensity.shape, (2,))
        self.assertEqual(learner.covariance.shape, (2, 2))
        self.assertEqual(learner.skewness.shape, (2, 2))
        self.assertTrue(np.isfinite(learner.objective()))
        self.assertGreaterEqual(learner.approximate_optimal_cs_ratio(), 0.0)
        self.assertLessEqual(learner.approximate_optimal_cs_ratio(), 1.0)
        self.assertEqual(learner.starting_point().shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "kernel norms only"):
            learner.get_kernel_values(0, 0, np.array([0.1, 0.2]))

    def test_parametric_learner_plotting_helpers(self):
        learner = HawkesExpKern(1.0, max_iter=1).fit(self.events, end_times=self.end_time)
        intensity_fig = learner.plot_estimated_intensity(self.events, intensity_track_step=0.1, show=False)
        qq_fig = learner.qq_plots(self.events, show=False)
        self.assertGreaterEqual(len(intensity_fig.axes), 1)
        self.assertGreaterEqual(len(qq_fig.axes), 1)

    def test_nonparametric_learner_objective_methods(self):
        em = HawkesEM(kernel_support=1.0, kernel_size=4, max_iter=1).fit(self.events, end_times=self.end_time)
        self.assertTrue(np.isfinite(em.objective(em.kernel)))


class TickPlotPublicBehaviorTest(unittest.TestCase):
    def test_hawkes_and_point_process_plot_smoke_tests(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        process = SimuPoissonProcess([0.5, 0.2], end_time=2.0, seed=42, verbose=False).simulate()
        process.store_compensator_values()
        point_fig = hawkes_plot.plot_point_process(process, show=False)
        qq_fig = hawkes_plot.qq_plots(process, show=False)

        events = [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])]
        learner = HawkesExpKern(1.0, gofit="likelihood", penalty="none", max_iter=2, verbose=False).fit(
            events, end_times=2.0
        )
        kernel_fig = hawkes_plot.plot_hawkes_kernels(learner, show=False)
        norms_fig = hawkes_plot.plot_hawkes_kernel_norms(learner, show=False)

        for fig in [point_fig, qq_fig, kernel_fig, norms_fig]:
            self.assertGreaterEqual(len(fig.axes), 1)
            plt.close(fig)

    def test_missing_tick_plot_helpers(self):
        self.assertTrue(hasattr(hawkes_plot, "plot_hawkes_baseline_and_kernels"))
        self.assertTrue(hasattr(hawkes_plot, "plot_basis_kernels"))
        self.assertTrue(hasattr(hawkes_plot, "plot_timefunction"))


class TickDocumentedGapCompatibilityTest(unittest.TestCase):
    def test_simu_hawkes_check_parameters_coherence_method(self):
        self.assertTrue(hasattr(SimuHawkes(n_nodes=1), "check_parameters_coherence"))

    @unittest.skip("our-hawkes parity gap: deep tick import paths such as tick.hawkes.simulation.base are not mirrored")
    def test_deep_tick_hawkes_import_paths(self):
        import our_hawkes.hawkes.simulation.base  # noqa: F401


if __name__ == "__main__":
    unittest.main()
