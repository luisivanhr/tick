import math
import sys
import unittest
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import (
    HawkesConditionalLaw,
    HawkesKernel0,
    HawkesKernelExp,
    HawkesKernelPowerLaw,
    HawkesKernelSumExp,
    ModelHawkesExpKernLeastSq,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik,
    SimuHawkesExpKernels,
)


def exponential_kernel(t, intensity, decay):
    return intensity * decay * np.exp(-decay * t)


def sum_exponential_kernel(t, intensities, decays):
    return sum(
        alpha * beta * np.exp(-beta * t)
        for alpha, beta in zip(intensities, decays)
    )


def hawkes_intensities(timestamps, baseline, kernels):
    dim = len(baseline)
    intensities = {}
    for i in range(dim):
        intensities[i] = lambda x, i=i: baseline[i] + sum(
            sum(kernels[i][j](x - timestamps[j][timestamps[j] < x]))
            for j in range(dim)
        )
    return intensities


def hawkes_exp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)
    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: exponential_kernel(
                t, adjacency[i, j], decays[i, j]
            )
    return hawkes_intensities(timestamps, baseline, kernels)


def hawkes_sumexp_kernel_intensities(baseline, decays, adjacency, timestamps):
    dim = len(timestamps)
    kernels = {}
    for i in range(dim):
        kernels[i] = {}
        for j in range(dim):
            kernels[i][j] = lambda t, i=i, j=j: sum_exponential_kernel(
                t, adjacency[i, j], decays
            )
    return hawkes_intensities(timestamps, baseline, kernels)


def hawkes_log_likelihood(intensities, timestamps, end_time, precision=4):
    dim = len(timestamps)
    compensator = sum(
        quad(
            lambda x, i=i: intensities[i](x),
            0,
            end_time,
            epsabs=10.0**-precision,
            limit=1000,
        )[0]
        for i in range(dim)
    )
    log_intensity = sum(
        sum(np.log(intensities[i](t)) for t in timestamps[i]) for i in range(dim)
    )
    return dim * end_time - compensator + log_intensity


def hawkes_least_square_error(intensities, timestamps, end_time, precision=4):
    dim = len(timestamps)
    squared_intensity_integral = sum(
        quad(
            lambda x, i=i: intensities[i](x) ** 2,
            0,
            end_time,
            epsabs=10.0**-precision,
            limit=1000,
        )[0]
        for i in range(dim)
    )
    intensity_convolution = sum(
        sum(intensities[i](t) for t in timestamps[i]) for i in range(dim)
    )
    return squared_intensity_integral - 2 * intensity_convolution


class TickKernelCompatibilityTest(unittest.TestCase):
    def test_exp_kernel_tick_strings_and_plot_support(self):
        kernel = HawkesKernelExp(3, 2)
        self.assertEqual(str(kernel), "3 * 2 * exp(- 2 * t)")
        self.assertEqual(str([kernel]), "[3*2*exp(-2*t)]")
        self.assertEqual(kernel.__strtex__(), "$6 e^{-2 t}$")
        self.assertEqual(kernel.get_plot_support(), 1.5)

        self.assertEqual(str(HawkesKernelExp(3, 0)), "3")
        self.assertEqual(HawkesKernelExp(3, 0).__strtex__(), "$3$")
        self.assertEqual(HawkesKernelExp(0.5, 2).__strtex__(), "$e^{-2 t}$")
        self.assertEqual(HawkesKernelExp(1, 1).__strtex__(), "$e^{-t}$")

    def test_sumexp_and_powerlaw_tick_strings(self):
        decays = np.array([1.0, 2.0, 0.2])
        intensities = np.array([0.3, 4.0, 2.0])
        kernel = HawkesKernelSumExp(intensities, decays)
        self.assertEqual(
            str(kernel),
            "0.3 * 1 * exp(- 1 * t) + 4 * 2 * exp(- 2 * t) + "
            "2 * 0.2 * exp(- 0.2 * t)",
        )
        self.assertEqual(
            kernel.__strtex__(), "$0.3 e^{- t}$ + $8 e^{-2 t}$ + $0.4 e^{-0.2 t}$"
        )

        self.assertEqual(HawkesKernelPowerLaw(0.1, 0.01, 0).__strtex__(), "$0.1$")
        self.assertEqual(
            HawkesKernelPowerLaw(1, 0.01, 1.2).__strtex__(),
            "$(0.01+t)^{-1.2}$",
        )


class TickSimulationCompatibilityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(23982)
        self.n_nodes = 3
        self.baseline = np.random.rand(self.n_nodes)
        self.adjacency = np.random.rand(self.n_nodes, self.n_nodes) / 2
        self.decays = np.random.rand(self.n_nodes, self.n_nodes)
        self.adjacency[0, 0] = 0
        self.adjacency[-1, -1] = 0
        self.hawkes = SimuHawkesExpKernels(
            self.adjacency,
            self.decays,
            baseline=self.baseline,
            seed=203,
            verbose=False,
        )

    def test_exp_kernel_construction_spectral_radius_and_mean(self):
        zero_kernel = None
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            kernel = self.hawkes.kernels[i, j]
            if self.adjacency[i, j] == 0:
                self.assertEqual(kernel.__class__, HawkesKernel0)
                zero_kernel = kernel if zero_kernel is None else zero_kernel
                self.assertIs(zero_kernel, kernel)
            else:
                self.assertEqual(kernel.__class__, HawkesKernelExp)
                self.assertEqual(kernel.decay, self.decays[i, j])
                self.assertEqual(kernel.intensity, self.adjacency[i, j])

        self.assertAlmostEqual(self.hawkes.spectral_radius(), 0.6645446549735008)
        self.hawkes.adjust_spectral_radius(0.6)
        self.assertAlmostEqual(self.hawkes.spectral_radius(), 0.6)
        expected_mean = np.linalg.solve(
            np.eye(self.n_nodes) - self.hawkes.adjacency, self.baseline
        )
        np.testing.assert_allclose(self.hawkes.mean_intensity(), expected_mean)

    def test_exp_compensator_matches_kernel_primitives(self):
        hawkes = SimuHawkesExpKernels(
            np.array([[0.3, 0.2], [0.1, 0.4]]),
            np.array([[1.5, 2.0], [0.7, 1.2]]),
            baseline=np.array([0.4, 0.2]),
            verbose=False,
        )
        timestamps = [np.array([0.2, 0.9, 1.8]), np.array([0.5, 1.1])]
        hawkes.set_timestamps(timestamps, end_time=2.0)
        hawkes.store_compensator_values()

        for i, compensators in enumerate(hawkes.tracked_compensator):
            expected = []
            for t in timestamps[i]:
                value = hawkes.baseline[i] * t
                for j in range(hawkes.n_nodes):
                    value += sum(
                        hawkes.adjacency[i, j]
                        * (1.0 - math.exp(-hawkes.decays[i, j] * (t - tj)))
                        for tj in timestamps[j]
                        if tj < t
                    )
                expected.append(value)
            np.testing.assert_allclose(compensators, expected, rtol=1e-12, atol=1e-12)


class TickModelCompatibilityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(30732)
        self.n_nodes = 3
        self.n_realizations = 2
        self.timestamps_list = [
            [
                np.cumsum(np.random.random(np.random.randint(3, 7)))
                for _ in range(self.n_nodes)
            ]
            for _ in range(self.n_realizations)
        ]

    def test_exp_loglik_loss_grad_hessian_norm_and_decay_change(self):
        decay = np.random.rand()
        baseline = np.random.rand(self.n_nodes)
        adjacency = np.random.rand(self.n_nodes, self.n_nodes)
        coeffs = np.hstack((baseline, adjacency.ravel()))

        model = ModelHawkesExpKernLogLik(decay).fit(
            self.timestamps_list[0], end_times=10.0
        )
        decays = np.ones((self.n_nodes, self.n_nodes)) * decay
        intensities = hawkes_exp_kernel_intensities(
            baseline, decays, adjacency, self.timestamps_list[0]
        )
        expected_loss = -hawkes_log_likelihood(
            intensities, self.timestamps_list[0], 10.0
        ) / model.n_jumps
        self.assertAlmostEqual(model.loss(coeffs), expected_loss, places=4)
        self.assertLess(check_grad(model.loss, model.grad, coeffs), 1e-5)

        hessian_point = np.random.rand(model.n_coeffs)
        vector = np.random.rand(model.n_coeffs)
        finite_diff = vector.dot(
            model.grad(hessian_point + 1e-7 * vector)
            - model.grad(hessian_point - 1e-7 * vector)
        ) / (2e-7)
        self.assertAlmostEqual(model.hessian_norm(hessian_point, vector), finite_diff, places=4)

        model_list = ModelHawkesExpKernLogLik(decay).fit(self.timestamps_list)
        changed = ModelHawkesExpKernLogLik(decay + 0.2).fit(self.timestamps_list)
        old_loss = changed.loss(coeffs)
        changed.decay = decay
        self.assertNotEqual(old_loss, changed.loss(coeffs))
        self.assertAlmostEqual(model_list.loss(coeffs), changed.loss(coeffs), places=12)

    def test_sumexp_loglik_loss_and_grad(self):
        decays = np.random.rand(2)
        baseline = np.random.rand(self.n_nodes)
        adjacency = np.random.rand(self.n_nodes, self.n_nodes, decays.size)
        coeffs = np.hstack((baseline, adjacency.ravel()))

        model = ModelHawkesSumExpKernLogLik(decays).fit(
            self.timestamps_list[0], end_times=10.0
        )
        intensities = hawkes_sumexp_kernel_intensities(
            baseline, decays, adjacency, self.timestamps_list[0]
        )
        expected_loss = -hawkes_log_likelihood(
            intensities, self.timestamps_list[0], 10.0
        ) / model.n_jumps
        self.assertAlmostEqual(model.loss(coeffs), expected_loss, places=4)
        self.assertLess(check_grad(model.loss, model.grad, coeffs), 1e-5)

    def test_exp_least_squares_loss_grad_and_expected_tick_values(self):
        timestamps = [
            np.array([0.2, 0.3, 0.65, 0.87, 1.0, 10.0, 12.0, 22.0]),
            np.array([3.0, 40.0, 60.0]),
        ]
        model = ModelHawkesExpKernLeastSq(decays=2.0).fit(timestamps)
        coeffs = np.array([0.1, 0.4, 0.3, 1.0, 0.4, 0.5])

        self.assertAlmostEqual(model.loss(coeffs), 1.05752053, delta=1e-8)
        np.testing.assert_allclose(
            model.grad(coeffs),
            np.array(
                [0.4363636, 4.5818182, -0.6009268, 0.4027132, 1.8310919, 0.3308908]
            ),
            atol=1e-7,
        )
        self.assertLess(check_grad(model.loss, model.grad, coeffs), 1e-5)

    def test_exp_and_sumexp_least_squares_match_quadrature(self):
        decays = np.random.rand(self.n_nodes, self.n_nodes)
        baseline = np.random.rand(self.n_nodes)
        adjacency = np.random.rand(self.n_nodes, self.n_nodes)
        coeffs = np.hstack((baseline, adjacency.ravel()))
        model = ModelHawkesExpKernLeastSq(decays=decays).fit(self.timestamps_list[0])
        intensities = hawkes_exp_kernel_intensities(
            baseline, decays, adjacency, self.timestamps_list[0]
        )
        expected = hawkes_least_square_error(
            intensities, self.timestamps_list[0], model.end_times[0]
        ) / model.n_jumps
        self.assertAlmostEqual(model.loss(coeffs), expected, places=4)
        self.assertLess(check_grad(model.loss, model.grad, coeffs), 1e-5)

        sum_decays = np.random.rand(2)
        sum_adjacency = np.random.rand(self.n_nodes, self.n_nodes, sum_decays.size)
        sum_coeffs = np.hstack((baseline, sum_adjacency.ravel()))
        sum_model = ModelHawkesSumExpKernLeastSq(decays=sum_decays).fit(
            self.timestamps_list[0]
        )
        sum_intensities = hawkes_sumexp_kernel_intensities(
            baseline, sum_decays, sum_adjacency, self.timestamps_list[0]
        )
        sum_expected = hawkes_least_square_error(
            sum_intensities, self.timestamps_list[0], sum_model.end_times[0]
        ) / sum_model.n_jumps
        self.assertAlmostEqual(sum_model.loss(sum_coeffs), sum_expected, places=4)
        self.assertLess(check_grad(sum_model.loss, sum_model.grad, sum_coeffs), 1e-5)


class TickConditionalLawCompatibilityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(320982)
        self.timestamps = [
            np.cumsum(np.random.random(np.random.randint(20, 25))) * 10.0
            for _ in range(2)
        ]

    def test_conditional_law_estimates_kernel_arrays_and_norms(self):
        learner = HawkesConditionalLaw(
            n_quad=8,
            max_support=10.0,
            min_support=1e-3,
            quad_method="gauss",
        ).fit(self.timestamps)

        self.assertEqual(learner.mean_intensity.shape, (2,))
        self.assertEqual(learner.baseline.shape, (2,))
        self.assertEqual(learner.kernels_norms.shape, (2, 2))
        self.assertEqual(len(learner.kernels), 2)
        self.assertEqual(len(learner.kernels[0]), 2)
        self.assertEqual(learner.kernels[0][1].shape, (2, learner.n_quad))
        self.assertTrue(np.all(np.isfinite(learner.get_kernel_norms())))
        self.assertTrue(np.any(np.abs(learner.get_kernel_norms()) > 0))

        values = learner.get_kernel_values(0, 1, np.array([-1.0, 0.5, 5.0, 20.0]))
        self.assertEqual(values.shape, (4,))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 0.0)

        supports = learner.get_kernel_supports()
        self.assertEqual(supports.shape, (2, 2))
        self.assertTrue(np.all(supports > 0.0))

    def test_conditional_law_incremental_compute_warning(self):
        learner = HawkesConditionalLaw(n_quad=6, max_support=5.0, min_support=1e-3)
        learner.incremental_fit(self.timestamps, compute=False)
        self.assertFalse(learner._has_been_computed_once())
        learner.compute()
        self.assertTrue(learner._has_been_computed_once())
        with self.assertWarnsRegex(UserWarning, "compute\\(\\) method was already called"):
            learner.incremental_fit(self.timestamps, compute=True)

    def test_conditional_law_quadrature_modes_and_symmetries(self):
        for method in ["gauss", "gauss-", "lin", "log"]:
            learner = HawkesConditionalLaw(
                n_quad=5,
                max_support=8.0,
                min_support=1e-3,
                quad_method=method,
                model={"symmetries1d": [[0, 1]], "symmetries2d": [[(0, 1), (1, 0)]]},
            ).fit(self.timestamps)
            self.assertTrue(np.all(np.isfinite(learner.mean_intensity)))
            self.assertTrue(np.all(np.isfinite(learner.baseline)))
            self.assertTrue(np.all(np.isfinite(learner.get_kernel_norms())))
            self.assertAlmostEqual(learner.mean_intensity[0], learner.mean_intensity[1])
            np.testing.assert_allclose(
                learner.get_kernel_values(0, 1, np.linspace(0.1, 5.0, 4)),
                learner.get_kernel_values(1, 0, np.linspace(0.1, 5.0, 4)),
            )


if __name__ == "__main__":
    unittest.main()
