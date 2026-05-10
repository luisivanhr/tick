import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.random import randint, random

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.hawkes import HawkesConditionalLaw


class TickConditionalLawParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(320982)
        self.timestamps = [
            np.cumsum(random(randint(20, 25))) * 10.0 for _ in range(2)
        ]

    def _fit_default(self):
        return HawkesConditionalLaw(n_quad=5).fit(self.timestamps)

    def test_tick_conditional_law_norm(self):
        model = self._fit_default()
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]],
        )

    def test_tick_conditional_law_kernels_fixture(self):
        model = self._fit_default()
        saved_phi_path = (
            Path(__file__).resolve().parents[3]
            / "tick"
            / "hawkes"
            / "inference"
            / "tests"
            / "hawkes_conditional_law_test-kernels.npy"
        )
        saved_phi = np.load(saved_phi_path)
        np.testing.assert_array_almost_equal(np.asarray(model.kernels), saved_phi)

    def test_tick_conditional_law_baseline(self):
        model = self._fit_default()
        np.testing.assert_array_almost_equal(model.baseline, [0.61213243, 0.808886425])

    def test_tick_conditional_mean_intensity(self):
        model = self._fit_default()
        np.testing.assert_array_almost_equal(model.mean_intensity, [0.208121177, 0.208121177])

    def test_tick_conditional_quad_methods(self):
        expected_by_method = {
            "gauss": [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]],
            "gauss-": [[-77.76904711, 0.69985519], [-42.87140913, 0.13607425]],
            "lin": [[7.92561315, 1.74540188], [-28.57048537, 10.77926367]],
        }
        for method, expected in expected_by_method.items():
            with self.subTest(method=method):
                model = HawkesConditionalLaw(n_quad=5, quad_method=method).fit(self.timestamps)
                np.testing.assert_array_almost_equal(model.kernels_norms, expected)

        model = HawkesConditionalLaw(n_quad=5, quad_method="log").fit(self.timestamps)
        np.testing.assert_allclose(
            model.kernels_norms,
            [[35.70738975, 18.96902121], [-51.69638233, -30.33936597]],
            atol=3e-3,
        )

    def test_tick_conditional_claw_methods(self):
        model = HawkesConditionalLaw(n_quad=5, claw_method="lin")
        model.incremental_fit(self.timestamps, compute=False)
        model.compute()
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]],
        )

        model = HawkesConditionalLaw(n_quad=5, claw_method="log")
        model.incremental_fit(self.timestamps)
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[0.46108403, -0.09467477], [-0.04787463, -3.82917571]],
        )

    def test_tick_conditional_incremental_fit_warnings(self):
        model = self._fit_default()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.incremental_fit(self.timestamps, compute=False)
        self.assertEqual(caught, [])

        msg = "compute\\(\\) method was already called, computed kernels will be updated."
        with self.assertWarnsRegex(UserWarning, msg):
            model.incremental_fit(self.timestamps, compute=True)

        new_model = HawkesConditionalLaw(n_quad=5, claw_method="lin")
        new_model.incremental_fit(self.timestamps, compute=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            new_model.incremental_fit(self.timestamps, compute=True)
        self.assertEqual(caught, [])
        with self.assertWarnsRegex(UserWarning, msg):
            new_model.incremental_fit(self.timestamps, compute=True)


if __name__ == "__main__":
    unittest.main()
