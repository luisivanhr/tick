"""Source-backed parity checks for tick's SumGaussians and BasisKernels tests.

The assertions in this file are ported from:

- tick/hawkes/inference/tests/hawkes_sumgaussians_test.py
- tick/hawkes/inference/tests/hawkes_basis_kernels_test.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.hawkes import HawkesBasisKernels, HawkesSumGaussians  # noqa: E402


def _toy_events():
    return [
        [
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33]),
        ],
        [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19]),
        ],
    ]


class HawkesSumGaussiansParityTest(unittest.TestCase):
    def setUp(self):
        self.int_1 = 4
        self.int_2 = 6
        self.float_1 = 0.3
        self.float_2 = 0.2

    def test_hawkes_sumgaussians_solution(self):
        events = _toy_events()
        n_nodes = len(events[0])
        n_gaussians = 3
        baseline_start = np.zeros(n_nodes) + 0.2
        amplitudes_start = np.zeros((n_nodes, n_nodes, n_gaussians)) + 0.2

        learner = HawkesSumGaussians(
            n_gaussians=n_gaussians,
            max_mean_gaussian=5,
            step_size=1e-3,
            C=10,
            lasso_grouplasso_ratio=0.7,
            n_threads=3,
            max_iter=11,
            verbose=False,
            em_max_iter=3,
        )
        learner.fit(events[0], baseline_start=baseline_start, amplitudes_start=amplitudes_start)

        np.testing.assert_array_almost_equal(learner.baseline, np.array([0.0979586, 0.15552228]), decimal=6)
        np.testing.assert_array_almost_equal(
            learner.amplitudes,
            np.array(
                [
                    [[0.20708954, -0.00627318, 0.08388442], [-0.00341803, 0.34805652, -0.00687372]],
                    [[-0.00341635, 0.1608013, 0.05531324], [-0.00342652, -0.00685425, 0.19046195]],
                ]
            ),
            decimal=6,
        )
        np.testing.assert_almost_equal(
            learner.get_kernel_values(0, 1, np.linspace(0, 4, 10)),
            np.array(
                [
                    -0.00068796,
                    0.01661161,
                    0.08872543,
                    0.21473618,
                    0.25597692,
                    0.15068586,
                    0.04194497,
                    0.00169372,
                    -0.00427233,
                    -0.00233042,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            learner.get_kernel_norms(),
            np.array([[0.28470077, 0.33776477], [0.21269818, 0.18018118]]),
        )
        np.testing.assert_array_almost_equal(learner.means_gaussians, np.array([0.0, 1.66666667, 3.33333333]))
        self.assertEqual(learner.std_gaussian, 0.5305164769729844)

        learner.n_gaussians += 1
        np.testing.assert_array_almost_equal(learner.means_gaussians, np.array([0.0, 1.25, 2.5, 3.75]))
        self.assertEqual(learner.std_gaussian, 0.3978873577297384)

    def test_hawkes_sumgaussians_set_data(self):
        events = _toy_events()
        learner = HawkesSumGaussians(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        learner = HawkesSumGaussians(1)
        learner._set_data(events[0])
        self.assertEqual(learner.n_nodes, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "All realizations should have 2 nodes, but realization 1 has 1 nodes",
        ):
            learner._set_data([events[0], [np.array([2, 3.2, 11.4, 12.8, 45])]])

    def test_hawkes_sumgaussians_parameters(self):
        learner = HawkesSumGaussians(1, n_gaussians=self.int_1)
        self.assertEqual(learner.n_gaussians, self.int_1)
        self.assertEqual(learner._learner.get_n_gaussians(), self.int_1)
        learner.n_gaussians = self.int_2
        self.assertEqual(learner.n_gaussians, self.int_2)
        self.assertEqual(learner._learner.get_n_gaussians(), self.int_2)

        learner = HawkesSumGaussians(max_mean_gaussian=self.float_1)
        self.assertEqual(learner.max_mean_gaussian, self.float_1)
        self.assertEqual(learner._learner.get_max_mean_gaussian(), self.float_1)
        learner.max_mean_gaussian = self.float_2
        self.assertEqual(learner.max_mean_gaussian, self.float_2)
        self.assertEqual(learner._learner.get_max_mean_gaussian(), self.float_2)

        learner = HawkesSumGaussians(1, step_size=self.float_1)
        self.assertEqual(learner.step_size, self.float_1)
        self.assertEqual(learner._learner.get_step_size(), self.float_1)
        learner.step_size = self.float_2
        self.assertEqual(learner.step_size, self.float_2)
        self.assertEqual(learner._learner.get_step_size(), self.float_2)

        with self.assertRaisesRegex(ValueError, "n_gaussians must be positive"):
            HawkesSumGaussians(1, n_gaussians=0)
        with self.assertRaisesRegex(ValueError, "max_mean_gaussian must be positive"):
            HawkesSumGaussians(0)
        with self.assertRaisesRegex(ValueError, "step_size must be positive"):
            HawkesSumGaussians(1, step_size=0)

    def test_hawkes_sumgaussians_lasso_grouplasso_ratio_parameter(self):
        C = 5e-3
        learner = HawkesSumGaussians(1, lasso_grouplasso_ratio=self.float_1, C=C)
        strength_lasso = self.float_1 / learner.C
        strength_grouplasso = (1.0 - self.float_1) / learner.C
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(), strength_grouplasso)
        self.assertEqual(learner.C, C)

        learner.lasso_grouplasso_ratio = self.float_2
        strength_lasso = self.float_2 / learner.C
        strength_grouplasso = (1.0 - self.float_2) / learner.C
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(), strength_grouplasso)
        self.assertEqual(learner.lasso_grouplasso_ratio, self.float_2)
        self.assertEqual(learner.C, C)

        with self.assertRaisesRegex(ValueError, "`lasso_grouplasso_ratio` must be between 0 and 1"):
            HawkesSumGaussians(1, lasso_grouplasso_ratio=1.1)

    def test_hawkes_sumgaussians_C_parameter(self):
        lasso_grouplasso_ratio = 0.3
        learner = HawkesSumGaussians(1, C=self.float_1, lasso_grouplasso_ratio=lasso_grouplasso_ratio)
        strength_lasso = learner.lasso_grouplasso_ratio / self.float_1
        strength_grouplasso = (1.0 - learner.lasso_grouplasso_ratio) / self.float_1
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(), strength_grouplasso)
        self.assertEqual(learner.lasso_grouplasso_ratio, lasso_grouplasso_ratio)

        learner.C = self.float_2
        strength_lasso = learner.lasso_grouplasso_ratio / self.float_2
        strength_grouplasso = (1.0 - learner.lasso_grouplasso_ratio) / self.float_2
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(), strength_grouplasso)
        self.assertAlmostEqual(learner.C, self.float_2)
        self.assertEqual(learner.lasso_grouplasso_ratio, lasso_grouplasso_ratio)

        with self.assertRaisesRegex(ValueError, "`C` must be positive"):
            HawkesSumGaussians(1, C=0)


class HawkesBasisKernelsParityTest(unittest.TestCase):
    def test_em_basis_kernels(self):
        ticks = _toy_events()
        n_basis = 2
        n_nodes = len(ticks[0])
        kernel_support = 4
        kernel_dt = 0.1
        kernel_size = int(np.ceil(kernel_support / kernel_dt))
        C = 5e-2

        mu = np.zeros(n_nodes) + 0.2
        auvd = np.zeros((n_nodes, n_nodes, n_basis)) + 0.4
        auvd[1, :, :] += 0.2
        gdm = np.zeros((n_basis, kernel_size))
        for i in range(kernel_size):
            gdm[0, i] = 0.1 * 0.29 * np.exp(-0.29 * i * kernel_dt)
            gdm[1, i] = 0.8 * np.exp(-i * kernel_dt)

        em = HawkesBasisKernels(
            kernel_support=kernel_support,
            kernel_size=kernel_size,
            n_basis=n_basis,
            C=C,
            n_threads=2,
            max_iter=5,
            ode_max_iter=100,
        )
        em.fit(ticks, baseline_start=mu, amplitudes_start=auvd, basis_kernels_start=gdm)

        np.testing.assert_array_almost_equal(em.baseline, [0.153022, 0.179124], decimal=4)
        np.testing.assert_array_almost_equal(
            em.amplitudes,
            [
                [[1.21125e-05, 1.744123e-03], [2.267314e-05, 3.287014e-03]],
                [[1.48773260e-05, 2.06898364e-03], [6.60131078e-06, 7.28397551e-04]],
            ],
            decimal=4,
        )

        basis_kernels = np.array(
            [
                [
                    0.0001699,
                    0.00031211,
                    0.00043944,
                    0.0005521,
                    0.00066688,
                    0.00078411,
                    0.0009040,
                    0.00101736,
                    0.001112,
                    0.00119935,
                    0.00129047,
                    0.00135828,
                    0.0014302,
                    0.00146572,
                    0.00149012,
                    0.00150987,
                    0.00152401,
                    0.00153267,
                    0.0015464,
                    0.00156525,
                    0.00157363,
                    0.00156589,
                    0.00156298,
                    0.00155548,
                    0.0015339,
                    0.00149196,
                    0.0014178,
                    0.0013323,
                    0.00125075,
                    0.00117292,
                    0.0010985,
                    0.00100652,
                    0.00091741,
                    0.00082029,
                    0.00071975,
                    0.00062118,
                    0.0005242,
                    0.0004228,
                    0.00029559,
                    0.00015301,
                ],
                [
                    0.0036240,
                    0.0066125,
                    0.00929557,
                    0.01163643,
                    0.01404666,
                    0.01653209,
                    0.0190978,
                    0.0215138,
                    0.02351321,
                    0.02535836,
                    0.02730293,
                    0.02874743,
                    0.0302991,
                    0.03096259,
                    0.03135936,
                    0.0316486,
                    0.03182045,
                    0.03187917,
                    0.0320631,
                    0.03237183,
                    0.03243794,
                    0.03216082,
                    0.03200454,
                    0.0317527,
                    0.0311904,
                    0.03017171,
                    0.02846015,
                    0.02650652,
                    0.02465883,
                    0.02291335,
                    0.0212655,
                    0.01931795,
                    0.01745896,
                    0.01547088,
                    0.01346942,
                    0.01154156,
                    0.0096817,
                    0.00779345,
                    0.00543011,
                    0.00279355,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(em.basis_kernels, basis_kernels, decimal=3)


if __name__ == "__main__":
    unittest.main()
