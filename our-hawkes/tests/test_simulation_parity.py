import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.base import TimeFunction
from our_hawkes.hawkes import (
    HawkesKernelExp,
    SimuHawkes,
    SimuHawkesExpKernels,
    SimuHawkesMulti,
    SimuHawkesSumExpKernels,
    SimuInhomogeneousPoisson,
    SimuPoissonProcess,
)


def assert_timestamps_equal(testcase, left, right):
    testcase.assertEqual(len(left), len(right))
    for actual, expected in zip(left, right):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


class SimulationParityTest(unittest.TestCase):
    def test_poisson_set_timestamps_compensator_and_seed_reproducibility(self):
        process = SimuPoissonProcess(2.5, verbose=False)
        self.assertEqual(process.intensities, 2.5)
        process.set_timestamps([np.array([0.2, 0.7, 1.1])], end_time=1.5)
        process.store_compensator_values()
        np.testing.assert_allclose(process.tracked_compensator[0], [0.5, 1.75, 2.75])

        first = SimuPoissonProcess([0.5, 0.2], end_time=5.0, seed=123, verbose=False).simulate()
        second = SimuPoissonProcess([0.5, 0.2], end_time=5.0, seed=123, verbose=False).simulate()
        assert_timestamps_equal(self, first.timestamps, second.timestamps)

        second.seed = 123
        second.reset()
        second.simulate()
        assert_timestamps_equal(self, first.timestamps, second.timestamps)

    def test_inhomogeneous_tracking_and_compensator_match_time_functions(self):
        times = np.linspace(0.0, 3.0, 7)
        values = np.array([0.0, 0.4, 0.8, 0.6, 0.3, 0.2, 0.0])
        tf = TimeFunction((times, values))
        process = SimuInhomogeneousPoisson([tf], end_time=3.0, seed=77, verbose=False)
        process.track_intensity(0.25)
        process.simulate()

        tracked_times = process.intensity_tracked_times
        np.testing.assert_allclose(process.tracked_intensity[0], tf.value(tracked_times), atol=1e-12)

        manual = SimuInhomogeneousPoisson([tf], verbose=False)
        manual.set_timestamps([np.array([0.5, 1.5, 2.5])], end_time=3.0)
        manual.store_compensator_values()
        np.testing.assert_allclose(
            manual.tracked_compensator[0],
            [tf.primitive(t) for t in [0.5, 1.5, 2.5]],
            atol=1e-12,
        )

    def test_hawkes_set_timestamps_tracks_history_and_resume_preserves_prefix(self):
        hawkes = SimuHawkesExpKernels(
            [[0.2, 0.1], [0.0, 0.3]],
            0.5,
            baseline=[0.2, 0.4],
            seed=1393,
            verbose=False,
        )
        original_timestamps = [
            np.array([7.096244, 9.389927]),
            np.array([0.436199, 0.659153, 2.622352, 3.095093, 7.189881, 8.068153, 9.240032]),
        ]

        hawkes.track_intensity(1.0)
        hawkes.set_timestamps(original_timestamps, 10.0)

        np.testing.assert_allclose(
            hawkes.tracked_intensity[0][:10],
            [
                0.2,
                0.2447256,
                0.27988282,
                0.24845138,
                0.23549475,
                0.27078386,
                0.26749709,
                0.27473586,
                0.24532959,
                0.22749379,
            ],
            atol=1e-7,
        )
        self.assertAlmostEqual(hawkes.simulation_time, 10.0)

        hawkes.end_time = 20.0
        hawkes.simulate()
        for actual, expected in zip(hawkes.timestamps, original_timestamps):
            np.testing.assert_allclose(actual[: expected.size], expected)
        self.assertAlmostEqual(hawkes.simulation_time, 20.0)

    def test_hawkes_exp_and_sumexp_compensators_are_exact_on_manual_history(self):
        exp = SimuHawkesExpKernels(
            np.array([[0.3, 0.2], [0.1, 0.4]]),
            np.array([[1.5, 2.0], [0.7, 1.2]]),
            baseline=np.array([0.4, 0.2]),
            verbose=False,
        )
        timestamps = [np.array([0.2, 0.9, 1.8]), np.array([0.5, 1.1])]
        exp.set_timestamps(timestamps, end_time=2.0)
        exp.store_compensator_values()
        for i, compensators in enumerate(exp.tracked_compensator):
            expected = []
            for t in timestamps[i]:
                value = exp.baseline[i] * t
                for j in range(exp.n_nodes):
                    value += sum(
                        exp.adjacency[i, j]
                        * (1.0 - math.exp(-exp.decays[i, j] * (t - tj)))
                        for tj in timestamps[j]
                        if tj < t
                    )
                expected.append(value)
            np.testing.assert_allclose(compensators, expected, rtol=1e-12, atol=1e-12)

        adjacency = np.array([[[0.2, 0.1]]])
        decays = np.array([1.0, 3.0])
        sumexp = SimuHawkesSumExpKernels(adjacency, decays, baseline=[0.4], verbose=False)
        sumexp_timestamps = [np.array([0.5, 1.2, 1.8])]
        sumexp.set_timestamps(sumexp_timestamps, end_time=2.0)
        sumexp.store_compensator_values()
        expected = []
        for t in sumexp_timestamps[0]:
            value = 0.4 * t
            value += sum(
                alpha * (1.0 - math.exp(-decay * (t - tj)))
                for tj in sumexp_timestamps[0]
                if tj < t
                for alpha, decay in zip(adjacency[0, 0], decays)
            )
            expected.append(value)
        np.testing.assert_allclose(sumexp.tracked_compensator[0], expected, rtol=1e-12, atol=1e-12)

    def test_spectral_radius_adjustment_and_negative_intensity_thresholding(self):
        hawkes = SimuHawkesExpKernels(
            [[0.0, 0.2], [0.1, 0.3]],
            1.5,
            baseline=[0.3, 0.4],
            verbose=False,
        )
        expected_radius = max(np.linalg.eigvals(np.array([[0.0, 0.2], [0.1, 0.3]])).real)
        self.assertAlmostEqual(hawkes.spectral_radius(), expected_radius)
        hawkes.adjust_spectral_radius(0.5)
        self.assertAlmostEqual(hawkes.spectral_radius(), 0.5)

        inhibitory = SimuHawkes(n_nodes=1, end_time=40.0, seed=1398, verbose=False)
        inhibitory.set_kernel(0, 0, HawkesKernelExp(-1.3, 0.8))
        inhibitory.set_baseline(0, 0.3)
        self.assertLess(inhibitory.spectral_radius(), 1.0)
        with self.assertRaisesRegex(RuntimeError, "intensity went negative"):
            inhibitory.simulate()

        clipped = SimuHawkes(n_nodes=1, end_time=40.0, seed=1398, verbose=False)
        clipped.set_kernel(0, 0, HawkesKernelExp(-1.3, 0.8))
        clipped.set_baseline(0, 0.3)
        clipped.threshold_negative_intensity()
        clipped.track_intensity(0.1)
        clipped.simulate()
        self.assertGreater(clipped.n_total_jumps, 1)
        self.assertGreaterEqual(clipped.tracked_intensity[0].min(), 0.0)
        self.assertLessEqual(clipped.tracked_intensity[0].max(), clipped.baseline[0])

    def test_hawkes_multi_reseeds_children_and_reproduces_realizations(self):
        def build_base(seed=504):
            return SimuHawkesExpKernels(
                [[0.2, 0.05], [0.1, 0.15]],
                1.2,
                baseline=[0.8, 0.6],
                end_time=20.0,
                seed=seed,
                verbose=False,
            )

        first = SimuHawkesMulti(build_base(), n_simulations=4, n_threads=1).simulate()
        second = SimuHawkesMulti(build_base(), n_simulations=4, n_threads=1).simulate()

        child_seeds = [first.get_single_simulation(i).seed for i in range(4)]
        self.assertEqual(len(set(child_seeds)), 4)
        self.assertEqual(first.simulation_time, [20.0] * 4)
        self.assertEqual(first.end_time, [20.0] * 4)
        self.assertEqual(first.n_nodes, [2] * 4)
        self.assertTrue(all(radius == first.spectral_radius[0] for radius in first.spectral_radius))

        self.assertEqual(first.n_total_jumps, second.n_total_jumps)
        for realization_a, realization_b in zip(first.timestamps, second.timestamps):
            assert_timestamps_equal(self, realization_a, realization_b)

        first.end_time = [25.0, 24.0, 23.0, 22.0]
        self.assertEqual(first.end_time, [25.0, 24.0, 23.0, 22.0])
        with self.assertRaisesRegex(ValueError, "end_time must have length 4"):
            first.end_time = [30.0]

        threaded = SimuHawkesMulti(build_base(), n_simulations=3, n_threads=2).simulate()
        self.assertEqual(threaded.simulation_time, [20.0] * 3)
        self.assertEqual(threaded.n_nodes, [2] * 3)
        self.assertTrue(all(np.isfinite(count) for count in threaded.n_total_jumps))


if __name__ == "__main__":
    unittest.main()
