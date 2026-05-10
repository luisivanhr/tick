import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesConditionalLaw


class ConditionalLawMarkedCompatibilityTest(unittest.TestCase):
    def test_marked_components_expand_intervals_and_estimate_mark_functions(self):
        events = [
            (
                np.array([0.1, 0.4, 0.9, 1.4]),
                np.array([0.5, 2.0, 4.5, 8.5]),
            ),
            np.array([0.2, 0.7, 1.2, 1.6]),
        ]

        learner = HawkesConditionalLaw(
            marked_components={0: [1.0, 3.0]},
            n_quad=4,
            max_support=1.0,
            max_lag=2.0,
            delta_lag=0.2,
        ).fit(events, T=2.0)

        self.assertEqual(len(learner.marked_components[0]), 3)
        self.assertEqual(len(learner.marked_components[1]), 1)
        np.testing.assert_allclose(learner._mark_probabilities[0], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(learner._mark_probabilities[1], [1.0])
        self.assertAlmostEqual(learner._mark_min[0], 0.5)
        self.assertAlmostEqual(learner._mark_max[0], 4.0)

        self.assertEqual(learner.kernels[0][0].shape, (2, learner.n_quad))
        self.assertEqual(learner.kernels_norms.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(learner.baseline)))

        marked_x, marked_y = learner.mark_functions[0][0]
        self.assertEqual(marked_x.shape, (300,))
        self.assertEqual(marked_y.shape, (300,))
        self.assertTrue(np.all(np.isfinite(marked_x)))
        self.assertTrue(np.all(np.isfinite(marked_y)))
        self.assertAlmostEqual(marked_x[0], learner._mark_min[0])
        self.assertAlmostEqual(marked_x[-1], learner._mark_max[0])

        unmarked_x, unmarked_y = learner.mark_functions[0][1]
        np.testing.assert_array_equal(unmarked_x, np.array([1.0]))
        np.testing.assert_array_equal(unmarked_y, np.array([1.0]))

    def test_delayed_components_shift_timestamps_and_preserve_marks(self):
        events = [
            (
                np.array([0.1, 0.4, 0.9]),
                np.array([1.0, 2.0, 3.0]),
            ),
            (
                np.array([0.1, 0.5]),
                np.array([2.0, 5.0]),
            ),
        ]

        learner = HawkesConditionalLaw(
            delayed_component=[1],
            delay=0.25,
            marked_components={1: [2.5]},
            n_quad=4,
            max_support=1.0,
        ).fit(events)

        np.testing.assert_allclose(learner.data[0][0], np.array([0.1, 0.4, 0.9]))
        np.testing.assert_allclose(learner.data[0][1], np.array([0.35, 0.75]))
        np.testing.assert_allclose(learner._marks[0][1], np.array([2.0, 5.0]))
        self.assertAlmostEqual(learner.end_times[0], 0.9)
        np.testing.assert_allclose(learner._mark_probabilities[1], [0.5, 0.5])

        with self.assertRaisesRegex(ValueError, "too small"):
            HawkesConditionalLaw(delayed_component=[1], delay=0.25).fit(events, T=0.6)

    def test_model_dict_and_set_model_apply_symmetries_and_delays(self):
        events = [
            (
                np.array([0.1, 0.3, 0.8, 1.4]),
                np.array([0.5, 1.5, 3.0, 5.0]),
            ),
            (
                np.array([0.2, 0.9, 1.1]),
                np.array([0.2, 1.4, 2.9]),
            ),
        ]
        model = {
            "symmetries1d": [(0, 1)],
            "symmetries2d": [((0, 0), (1, 1)), ((0, 1), (1, 0))],
            "delayed_component": [0],
        }

        learner = HawkesConditionalLaw(
            marked_components={0: [1.0], 1: [1.0]},
            delay=0.1,
            model=model,
            n_quad=5,
            max_support=1.2,
        ).fit(events)

        self.assertAlmostEqual(learner.mean_intensity[0], learner.mean_intensity[1])
        np.testing.assert_allclose(learner._mark_probabilities[0], learner._mark_probabilities[1])
        np.testing.assert_allclose(
            learner.get_kernel_values(0, 0, np.linspace(0.1, 1.0, 4)),
            learner.get_kernel_values(1, 1, np.linspace(0.1, 1.0, 4)),
        )
        np.testing.assert_allclose(
            learner.get_kernel_values(0, 1, np.linspace(0.1, 1.0, 4)),
            learner.get_kernel_values(1, 0, np.linspace(0.1, 1.0, 4)),
        )
        np.testing.assert_allclose(learner.data[0][0], np.array([0.2, 0.4, 0.9, 1.5]))

        learner.set_model(symmetries1d=[(0, 1)], delayed_component=None)
        self.assertIsNone(learner.delayed_component)

    def test_incremental_empty_realization_warns_and_compute_fails_cleanly(self):
        learner = HawkesConditionalLaw(n_quad=3)
        with self.assertWarnsRegex(UserWarning, "empty realization"):
            learner.incremental_fit([np.array([]), np.array([])], compute=False)
        self.assertEqual(learner.n_realizations, 0)
        with self.assertRaisesRegex(ValueError, "no realizations"):
            learner.compute()

    def test_validation_for_marked_inputs_and_options(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            HawkesConditionalLaw(marked_components={0: [2.0, 1.0]}).fit([np.array([0.1])])

        with self.assertRaisesRegex(ValueError, "marks must match"):
            HawkesConditionalLaw().fit([(np.array([0.1, 0.2]), np.array([1.0]))])

        with self.assertRaisesRegex(ValueError, "delayed_component"):
            HawkesConditionalLaw(delayed_component=[3]).fit([np.array([0.1]), np.array([0.2])])

        with self.assertRaisesRegex(ValueError, "claw_method"):
            HawkesConditionalLaw(claw_method="bad")

        with self.assertRaisesRegex(ValueError, "quad_method"):
            HawkesConditionalLaw(quad_method="bad")


if __name__ == "__main__":
    unittest.main()
