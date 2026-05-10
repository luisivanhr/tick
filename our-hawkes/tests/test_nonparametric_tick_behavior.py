import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesBasisKernels, HawkesEM, HawkesSumGaussians


class TickNonParametricLearnerBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.events = [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])]
        self.end_time = 2.0

    def test_em_kernel_discretization_setters_match_tick_style_grid_updates(self):
        learner = HawkesEM(kernel_support=4.4)
        self.assertEqual(learner.kernel_size, 10)
        self.assertEqual(learner.kernel_support, 4.4)
        self.assertAlmostEqual(learner.kernel_dt, 0.44)
        np.testing.assert_allclose(learner.kernel_discretization, np.linspace(0.0, 4.4, 11))

        learner.kernel_support = 6.2
        self.assertEqual(learner.kernel_size, 10)
        np.testing.assert_allclose(learner.kernel_discretization, np.linspace(0.0, 6.2, 11))

        learner.kernel_size = 5
        self.assertAlmostEqual(learner.kernel_dt, 1.24)
        np.testing.assert_allclose(learner.kernel_discretization, np.linspace(0.0, 6.2, 6))

        learner = HawkesEM(kernel_support=4.0, kernel_size=10)
        learner.kernel_dt = 0.199
        self.assertEqual(learner.kernel_size, 21)
        self.assertAlmostEqual(learner.kernel_dt, 4.0 / 21.0)

        non_uniform = HawkesEM(kernel_discretization=np.array([0.0, 1.0, 1.5, 3.0]))
        self.assertEqual(non_uniform.kernel_support, 3.0)
        self.assertEqual(non_uniform.kernel_size, 3)
        np.testing.assert_allclose(non_uniform.kernel_dt, np.array([1.0, 0.5, 1.5]))

    def test_em_score_and_time_change_can_use_explicit_parameters_without_fit(self):
        events = [np.array([0.5, 1.5])]
        baseline = np.array([2.0])
        kernel = np.zeros((1, 1, 2))
        learner = HawkesEM(kernel_support=1.0, kernel_size=2)

        with self.assertRaisesRegex(ValueError, "fit.*score|score.*fit"):
            learner.score()

        score = learner.score(events, end_times=2.0, baseline=baseline, kernel=kernel)
        expected = (2.0 - 2.0 * 2.0 + 2.0 * math.log(2.0)) / 2.0
        self.assertAlmostEqual(score, expected)

        residuals = learner.time_changed_interarrival_times(
            events, end_times=2.0, baseline=baseline, kernel=kernel
        )
        np.testing.assert_allclose(residuals[0][0], np.array([2.0]))

        fitted = HawkesEM(kernel_support=1.0, kernel_size=2, max_iter=0).fit(
            events, end_times=2.0, baseline_start=baseline, kernel_start=kernel
        )
        self.assertAlmostEqual(fitted.score(), expected)

    def test_em_kernel_accessors_norms_and_warm_start_copies(self):
        baseline_start = np.array([0.2, 0.3])
        kernel_start = np.arange(1, 13, dtype=float).reshape(2, 2, 3) / 10.0

        learner = HawkesEM(kernel_support=3.0, kernel_size=3, max_iter=0).fit(
            self.events,
            end_times=self.end_time,
            baseline_start=baseline_start,
            kernel_start=kernel_start,
        )
        baseline_start[0] = 99.0
        kernel_start[0, 0, 0] = 99.0

        np.testing.assert_allclose(learner.baseline, np.array([0.2, 0.3]))
        self.assertNotEqual(learner.kernel[0, 0, 0], 99.0)
        np.testing.assert_allclose(learner.get_kernel_supports(), np.full((2, 2), 3.0))

        x = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
        np.testing.assert_allclose(
            learner.get_kernel_values(0, 1, x),
            np.array([0.0, learner.kernel[0, 1, 0], learner.kernel[0, 1, 0], learner.kernel[0, 1, 2], 0.0]),
        )
        np.testing.assert_allclose(learner.get_kernel_norms(), learner.kernel.sum(axis=2))
        np.testing.assert_allclose(
            learner._compute_primitive_kernel_values(0, 1, np.array([1.0, 2.0, 3.0])),
            np.cumsum(learner.kernel[0, 1]),
        )

    def test_sumgaussians_accessors_regularization_and_warm_starts(self):
        baseline_start = np.array([0.2, 0.3])
        amplitudes_start = np.arange(1, 13, dtype=float).reshape(2, 2, 3) / 20.0

        learner = HawkesSumGaussians(
            max_mean_gaussian=5.0,
            n_gaussians=3,
            C=10.0,
            lasso_grouplasso_ratio=0.7,
            max_iter=0,
        ).fit(
            self.events,
            end_times=self.end_time,
            baseline_start=baseline_start,
            amplitudes_start=amplitudes_start,
        )
        baseline_start[0] = 99.0
        amplitudes_start[0, 0, 0] = 99.0

        self.assertAlmostEqual(learner.strength_lasso, 0.07)
        self.assertAlmostEqual(learner.strength_grouplasso, 0.03)
        learner.C = 5.0
        learner.lasso_grouplasso_ratio = 0.25
        self.assertAlmostEqual(learner.strength_lasso, 0.05)
        self.assertAlmostEqual(learner.strength_grouplasso, 0.15)

        np.testing.assert_allclose(learner.baseline, np.array([0.2, 0.3]))
        self.assertNotEqual(learner.amplitudes[0, 0, 0], 99.0)
        np.testing.assert_allclose(learner.get_kernel_supports(), np.full((2, 2), 3.0))
        np.testing.assert_allclose(learner.get_kernel_norms(), learner.amplitudes.sum(axis=2))

        x = np.linspace(0.0, 4.0, 5)
        sigma = learner.std_gaussian
        expected = np.zeros_like(x)
        for amplitude, mean in zip(learner.amplitudes[0, 1], learner.means_gaussians):
            expected += amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (
                sigma * np.sqrt(2.0 * np.pi)
            )
        np.testing.assert_allclose(learner.get_kernel_values(0, 1, x), expected)

    def test_basis_kernel_accessors_norms_discretization_and_warm_starts(self):
        baseline_start = np.array([0.2, 0.3])
        amplitudes_start = np.arange(1, 9, dtype=float).reshape(2, 2, 2) / 10.0
        basis_kernels_start = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])

        learner = HawkesBasisKernels(kernel_support=3.0, kernel_size=3, n_basis=2, max_iter=0).fit(
            self.events,
            end_times=self.end_time,
            baseline_start=baseline_start,
            amplitudes_start=amplitudes_start,
            basis_kernels_start=basis_kernels_start,
        )
        baseline_start[0] = 99.0
        amplitudes_start[0, 0, 0] = 99.0
        basis_kernels_start[0, 0] = 99.0

        np.testing.assert_allclose(learner.baseline, np.array([0.2, 0.3]))
        self.assertNotEqual(learner.amplitudes[0, 0, 0], 99.0)
        self.assertNotEqual(learner.basis_kernels[0, 0], 99.0)
        np.testing.assert_allclose(learner.get_kernel_supports(), np.full((2, 2), 3.0))

        learner.kernel_dt = 0.8
        self.assertEqual(learner.kernel_size, 4)
        np.testing.assert_allclose(learner.kernel_discretization, np.linspace(0.0, 3.0, 5))

        learner.kernel_size = 3
        x = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
        kernel_10 = learner.amplitudes[1, 0] @ learner.basis_kernels
        np.testing.assert_allclose(
            learner.get_kernel_values(1, 0, x),
            np.array([0.0, kernel_10[0], kernel_10[0], kernel_10[2], 0.0]),
        )

        basis_norms = learner.basis_kernels.sum(axis=1)
        expected_norms = np.tensordot(learner.amplitudes, basis_norms, axes=(2, 0))
        np.testing.assert_allclose(learner.get_kernel_norms(), expected_norms)

    def test_seeded_random_starts_are_reproducible(self):
        np.random.seed(2026)
        em_1 = HawkesEM(kernel_support=1.0, kernel_size=4, max_iter=1).fit(self.events, self.end_time)
        np.random.seed(2026)
        em_2 = HawkesEM(kernel_support=1.0, kernel_size=4, max_iter=1).fit(self.events, self.end_time)
        np.testing.assert_allclose(em_1.baseline, em_2.baseline)
        np.testing.assert_allclose(em_1.kernel, em_2.kernel)

        np.random.seed(2026)
        sg_1 = HawkesSumGaussians(1.0, n_gaussians=4, max_iter=0).fit(self.events, self.end_time)
        np.random.seed(2026)
        sg_2 = HawkesSumGaussians(1.0, n_gaussians=4, max_iter=0).fit(self.events, self.end_time)
        np.testing.assert_allclose(sg_1.baseline, sg_2.baseline)
        np.testing.assert_allclose(sg_1.amplitudes, sg_2.amplitudes)

        np.random.seed(2026)
        bk_1 = HawkesBasisKernels(1.0, kernel_size=4, max_iter=0).fit(self.events, self.end_time)
        np.random.seed(2026)
        bk_2 = HawkesBasisKernels(1.0, kernel_size=4, max_iter=0).fit(self.events, self.end_time)
        np.testing.assert_allclose(bk_1.baseline, bk_2.baseline)
        np.testing.assert_allclose(bk_1.amplitudes, bk_2.amplitudes)
        np.testing.assert_allclose(bk_1.basis_kernels, bk_2.basis_kernels)


if __name__ == "__main__":
    unittest.main()
