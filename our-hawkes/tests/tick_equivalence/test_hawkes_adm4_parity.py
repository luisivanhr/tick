"""Source-backed parity checks for tick's HawkesADM4 tests.

The assertions in this file are ported from
tick/hawkes/inference/tests/hawkes_adm4_test.py.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.hawkes import HawkesADM4, SimuHawkesExpKernels, SimuHawkesMulti  # noqa: E402


class HawkesADM4ParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(329832)
        self.decay = 0.7
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2

    def simulate_sparse_realization(self):
        baseline = np.array([0.3, 0.001])
        adjacency = np.array([[0.5, 0.8], [0.0, 1.3]])

        sim = SimuHawkesExpKernels(
            adjacency=adjacency,
            decays=self.decay,
            baseline=baseline,
            verbose=False,
            seed=13487,
            end_time=500,
        )
        sim.adjust_spectral_radius(0.8)
        multi = SimuHawkesMulti(sim, n_simulations=100)

        adjacency = sim.adjacency.copy()
        multi.simulate()

        self.assertGreater(max(map(lambda r: len(r[1]), multi.timestamps)), 1)
        self.assertEqual(min(map(lambda r: len(r[1]), multi.timestamps)), 0)

        return baseline, adjacency, multi.timestamps

    def test_sparse(self):
        baseline, adjacency, events = self.simulate_sparse_realization()
        learner = HawkesADM4(self.decay, verbose=False)
        learner.fit(events)

        np.testing.assert_array_almost_equal(learner.baseline, baseline, decimal=1)
        np.testing.assert_array_almost_equal(learner.adjacency, adjacency, decimal=1)

    def test_hawkes_adm4_solution(self):
        events = [
            [
                np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
                np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33]),
            ],
            [
                np.array([2, 3.2, 11.4, 12.8, 45]),
                np.array([2, 3, 8.8, 9, 15.3, 19]),
            ],
        ]

        learner = HawkesADM4(
            self.decay,
            rho=0.5,
            C=10,
            lasso_nuclear_ratio=0.7,
            n_threads=3,
            max_iter=11,
            verbose=False,
            em_max_iter=3,
            record_every=1,
        )
        learner.fit(
            events[0],
            baseline_start=np.zeros(2) + 0.2,
            adjacency_start=np.zeros((2, 2)) + 0.2,
        )

        np.testing.assert_array_almost_equal(
            learner.baseline, np.array([0.14551, 0.239859]), decimal=6
        )
        np.testing.assert_array_almost_equal(
            learner.adjacency,
            np.array([[2.275416e-01, 8.234672e-02], [1.195861e-02, 4.070548e-10]]),
            decimal=6,
        )

    def test_hawkes_adm4_score(self):
        n_nodes = 2
        n_realizations = 3

        train_events = [
            [np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)]
            for _ in range(n_realizations)
        ]
        test_events = [
            [np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)]
            for _ in range(n_realizations)
        ]

        learner = HawkesADM4(self.decay, record_every=1)
        with self.assertRaisesRegex(
            ValueError, r"^You must either call `fit` before `score` or provide events$"
        ):
            learner.score()

        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes)

        learner.fit(train_events)
        self.assertAlmostEqual(learner.score(), 0.12029826, places=6)
        self.assertAlmostEqual(
            learner.score(baseline=given_baseline, adjacency=given_adjacency),
            -0.15247511,
            places=6,
        )
        self.assertAlmostEqual(learner.score(test_events), 0.17640007, places=6)
        self.assertAlmostEqual(
            learner.score(test_events, baseline=given_baseline, adjacency=given_adjacency),
            -0.07973875,
            places=6,
        )

    def test_hawkes_adm4_set_data(self):
        events = [
            [
                np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
                np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33]),
            ],
            [
                np.array([2, 3.2, 11.4, 12.8, 45]),
                np.array([2, 3, 8.8, 9, 15.3, 19]),
            ],
        ]

        learner = HawkesADM4(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        learner = HawkesADM4(1)
        learner._set_data(events[0])
        self.assertEqual(learner.n_nodes, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "All realizations should have 2 nodes, but realization 1 has 1 nodes",
        ):
            learner._set_data([events[0], [np.array([2, 3.2, 11.4, 12.8, 45])]])

    def test_hawkes_adm4_parameters(self):
        learner = HawkesADM4(self.float_1)
        self.assertEqual(learner.decay, self.float_1)
        self.assertEqual(learner._learner.get_decay(), self.float_1)
        learner.decay = self.float_2
        self.assertEqual(learner.decay, self.float_2)
        self.assertEqual(learner._learner.get_decay(), self.float_2)

        learner = HawkesADM4(1, rho=self.float_1)
        self.assertEqual(learner.rho, self.float_1)
        self.assertEqual(learner._learner.get_rho(), self.float_1)
        learner.rho = self.float_2
        self.assertEqual(learner.rho, self.float_2)
        self.assertEqual(learner._learner.get_rho(), self.float_2)

        with self.assertRaisesRegex(ValueError, "decay must be positive"):
            HawkesADM4(0.0)
        with self.assertRaisesRegex(ValueError, "rho .* must be positive"):
            HawkesADM4(1.0, rho=0.0)
        with self.assertRaisesRegex(ValueError, "`C` must be positive"):
            HawkesADM4(1.0, C=-1)
        with self.assertRaisesRegex(ValueError, "`lasso_nuclear_ratio` must be between 0 and 1"):
            HawkesADM4(1.0, lasso_nuclear_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
