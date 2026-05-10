import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.base import TimeFunction
from our_hawkes.hawkes import (
    HawkesEM,
    HawkesKernelExp,
    HawkesKernelTimeFunc,
    SimuHawkes,
    SimuHawkesExpKernels,
)


def compute_approx_support_of_exp_kernel(adjacency, decays, eps):
    return np.maximum(0.0, np.squeeze(np.max(-np.log(eps / (adjacency * decays)) / decays)))


def discretization_of_exp_kernel(n_nodes, adjacency, decays, kernel_support, kernel_size):
    abscissa = np.linspace(0.0, kernel_support, kernel_size)
    abscissa = np.repeat(abscissa[None, :], repeats=n_nodes, axis=0)
    abscissa = np.repeat(abscissa[None, :, :], repeats=n_nodes, axis=0)
    adjacency = np.repeat(np.asarray(adjacency, dtype=float)[:, :, None], repeats=kernel_size, axis=2)
    decays = np.repeat(np.asarray(decays, dtype=float)[:, :, None], repeats=kernel_size, axis=2)
    return adjacency * decays * np.exp(-decays * abscissa)


def piecewise_loglik_reference(events, end_times, baseline, kernel, discretization):
    if isinstance(end_times, (int, float, np.floating)):
        end_times = [float(end_times)]
    if isinstance(events[0], np.ndarray):
        events = [events]
    value = 0.0
    n_jumps = 0
    n_nodes = len(events[0])
    for realization, end_time in zip(events, end_times):
        value += n_nodes * float(end_time)
        for i in range(n_nodes):
            value -= baseline[i] * float(end_time)
            for j in range(n_nodes):
                for tj in realization[j]:
                    if tj >= end_time:
                        break
                    remaining = float(end_time - tj)
                    for m in range(kernel.shape[2]):
                        overlap = max(
                            0.0,
                            min(remaining, discretization[m + 1]) - discretization[m],
                        )
                        value -= kernel[i, j, m] * overlap

            for t in realization[i]:
                n_jumps += 1
                intensity = baseline[i]
                for j in range(n_nodes):
                    for tj in realization[j]:
                        if tj >= t:
                            break
                        lag = float(t - tj)
                        if lag >= discretization[-1]:
                            continue
                        m = int(np.searchsorted(discretization, lag) - 1)
                        if 0 <= m < kernel.shape[2]:
                            intensity += kernel[i, j, m]
                if intensity <= 0.0:
                    return -np.inf
                value += np.log(intensity)
    return float(value / max(n_jumps, 1))


def simulate_small_exp_hawkes():
    decays = np.array([[1.0, 1.5], [0.1, 0.5]])
    baseline = np.array([0.12, 0.07])
    adjacency = np.array([[0.1, 0.4], [0.2, 0.03]])
    model = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=800,
        max_jumps=2000,
        verbose=False,
        force_simulation=False,
        seed=12345,
    )
    model.simulate()
    return model, baseline, adjacency, decays


def simulate_small_nonparam_hawkes():
    t_values1 = np.array([0.0, 1.0, 1.5, 2.0, 3.5])
    y_values1 = np.array([0.0, 0.2, 0.0, 0.1, 0.0])
    kernel1 = HawkesKernelTimeFunc(
        TimeFunction([t_values1, y_values1], inter_mode=TimeFunction.InterConstRight, dt=0.1)
    )

    t_values2 = np.linspace(0.0, 4.0, 20)
    y_values2 = np.maximum(0.0, np.sin(t_values2) / 4.0)
    kernel2 = HawkesKernelTimeFunc(TimeFunction([t_values2, y_values2]))

    hawkes = SimuHawkes(
        baseline=np.array([0.1, 0.3]),
        end_time=200,
        verbose=False,
        seed=2334,
    )
    hawkes.set_kernel(0, 0, kernel1)
    hawkes.set_kernel(0, 1, HawkesKernelExp(0.5, 0.7))
    hawkes.set_kernel(1, 1, kernel2)
    hawkes.simulate()
    return hawkes


class TickHawkesEMParityTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(123269)
        self.n_nodes = 3
        self.n_realizations = 2
        self.events = [
            [np.cumsum(np.random.rand(4 + i)) for i in range(self.n_nodes)]
            for _ in range(self.n_realizations)
        ]

    def test_hawkes_em_attributes(self):
        em = HawkesEM(kernel_support=10)
        em.fit(self.events)
        self.assertEqual(em.n_nodes, self.n_nodes)
        self.assertEqual(em.n_realizations, self.n_realizations)

    def test_hawkes_em_fit_1(self):
        baseline = np.zeros(self.n_nodes) + 0.2
        kernel = np.zeros((self.n_nodes, self.n_nodes, 3)) + 0.4

        em = HawkesEM(kernel_support=3, kernel_size=3, n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events, baseline_start=baseline, kernel_start=kernel)

        np.testing.assert_array_almost_equal(em.baseline, [1.2264, 0.2164, 1.6782], decimal=4)
        expected_kernel = [
            [[2.4569e-02, 2.5128e-06, 0.0], [1.8072e-02, 5.4332e-11, 0.0], [2.7286e-03, 4.0941e-08, 3.5705e-15]],
            [[8.0077e-01, 2.2624e-02, 6.7577e-10], [2.7503e-02, 3.1840e-05, 0.0], [1.4984e-01, 7.8428e-06, 2.8206e-12]],
            [[1.2163e-01, 1.0997e-02, 5.4724e-05], [4.7348e-02, 6.6093e-03, 5.5433e-12], [1.0662e-03, 5.3920e-05, 1.4930e-08]],
        ]
        np.testing.assert_array_almost_equal(em.kernel, expected_kernel, decimal=4)

        em2 = HawkesEM(kernel_discretization=np.array([0.0, 1.0, 2.0, 3.0]), n_threads=1, max_iter=11, verbose=False)
        em2.fit(self.events, baseline_start=baseline, kernel_start=kernel)
        np.testing.assert_array_almost_equal(em2.kernel, expected_kernel, decimal=4)

        np.testing.assert_array_almost_equal(
            em.get_kernel_values(1, 0, np.linspace(0.0, 3.0, 5)),
            [0.0, 8.0077e-01, 2.2624e-02, 6.7577e-10, 0.0],
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            em.get_kernel_norms(),
            [[0.0246, 0.0181, 0.0027], [0.8234, 0.0275, 0.1499], [0.1327, 0.054, 0.0011]],
            decimal=3,
        )
        np.testing.assert_array_equal(em.get_kernel_supports(), np.ones((self.n_nodes, self.n_nodes)) * 3)

    def test_hawkes_em_fit_2(self):
        simu, baseline, adjacency, decays = simulate_small_exp_hawkes()
        kernel_support = compute_approx_support_of_exp_kernel(adjacency, decays, 1e-4)
        kernel_size = 20
        baseline_start = np.array([0.05 * np.mean(np.diff(ts)) for ts in simu.timestamps])
        np.random.seed(123269)
        kernel_start = np.zeros((2, 2, kernel_size))
        kernel_start[:, :, : kernel_size - 1] = 0.01 * np.cumsum(
            np.random.uniform(size=(2, 2, kernel_size - 1)), axis=2
        )[:, :, ::-1]

        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size, tol=1e-5, max_iter=40, verbose=False)
        em.fit(simu.timestamps, baseline_start=baseline_start, kernel_start=kernel_start)

        self.assertEqual(em.baseline.shape, baseline.shape)
        np.testing.assert_array_equal(np.argsort(em.baseline), np.argsort(baseline))
        np.testing.assert_array_almost_equal(em.baseline, baseline, decimal=1)
        self.assertEqual(em.kernel.shape, (2, 2, kernel_size))
        self.assertTrue(np.all(em.kernel >= 0.0))
        significant = np.diff(em.kernel, append=0.0, axis=2)
        significant = significant[np.abs(significant) > 0.2]
        self.assertTrue(np.all(significant <= 0.0))

    def test_hawkes_em_fit_3(self):
        simu = simulate_small_nonparam_hawkes()
        em = HawkesEM(kernel_support=4.0, kernel_size=16, n_threads=2, verbose=False, max_iter=20, tol=1e-5)
        em.fit(simu.timestamps)

        evaluation_points = np.linspace(0.0, 4.0, num=6)
        for i in range(2):
            for j in range(2):
                estimated = em._compute_primitive_kernel_values(i, j, evaluation_points)
                expected = simu.kernels[i, j].get_primitive_values(evaluation_points)
                self.assertTrue(np.all(np.isfinite(estimated)))
                self.assertTrue(np.all(np.diff(estimated) >= -1e-12))
                np.testing.assert_allclose(estimated, expected, atol=0.18, rtol=2.5)

    def test_hawkes_em_score(self):
        n_nodes = 2
        baseline = np.random.rand(n_nodes) + 0.2
        kernel = np.random.rand(n_nodes, n_nodes, 3) + 0.4
        train_events = [np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]
        test_events = [2.0 + np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]

        for kwargs, fit in [
            ({"kernel_support": 1, "kernel_size": 3}, True),
            ({"kernel_discretization": np.array([0.0, 1.0, 1.5, 3.0])}, False),
        ]:
            em = HawkesEM(**kwargs)
            end_times = max(map(max, train_events)) + 0.2
            with self.assertRaisesRegex(ValueError, "^You must either call `fit` before `score` or provide events"):
                em.score()

            if fit:
                em.fit(train_events, end_times=end_times, baseline_start=baseline, kernel_start=kernel)
            else:
                em.baseline = baseline
                em.kernel = kernel

            expected_train = piecewise_loglik_reference(
                train_events, end_times, em.baseline, em.kernel, em.kernel_discretization
            )
            actual_train = em.score() if fit else em.score(train_events, end_times=end_times)
            self.assertAlmostEqual(actual_train, expected_train, places=12)

            test_end_times = max(map(max, test_events))
            expected_test = piecewise_loglik_reference(
                test_events, test_end_times, em.baseline, em.kernel, em.kernel_discretization
            )
            self.assertAlmostEqual(em.score(events=test_events), expected_test, places=12)

    def test_hawkes_em_kernel_shape(self):
        em = HawkesEM(kernel_support=4, kernel_size=10, n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)
        reshaped_kernel = em._flat_kernels.reshape((em.n_nodes, em.n_nodes, em.kernel_size))
        self.assertTrue(np.allclose(em.kernel, reshaped_kernel))

    def test_hawkes_em_kernel_support(self):
        learner = HawkesEM(4.4)
        self.assertEqual(learner.kernel_support, 4.4)
        self.assertEqual(learner._learner.get_kernel_support(), 4.4)
        np.testing.assert_array_almost_equal(
            learner.kernel_discretization,
            [0.0, 0.44, 0.88, 1.32, 1.76, 2.2, 2.64, 3.08, 3.52, 3.96, 4.4],
        )

        learner.kernel_support = 6.2
        self.assertEqual(learner.kernel_support, 6.2)
        self.assertEqual(learner._learner.get_kernel_support(), 6.2)
        np.testing.assert_array_almost_equal(
            learner.kernel_discretization,
            [0.0, 0.62, 1.24, 1.86, 2.48, 3.1, 3.72, 4.34, 4.96, 5.58, 6.2],
        )

    def test_hawkes_em_kernel_size(self):
        learner = HawkesEM(4.0, kernel_size=4)
        self.assertEqual(learner.kernel_size, 4)
        self.assertEqual(learner._learner.get_kernel_size(), 4)
        np.testing.assert_array_almost_equal(learner.kernel_discretization, [0.0, 1.0, 2.0, 3.0, 4.0])

        learner.kernel_size = 5
        self.assertEqual(learner.kernel_size, 5)
        self.assertEqual(learner._learner.get_kernel_size(), 5)
        np.testing.assert_array_almost_equal(learner.kernel_discretization, [0.0, 0.8, 1.6, 2.4, 3.2, 4.0])

    def test_hawkes_em_kernel_dt(self):
        learner = HawkesEM(4.0, kernel_size=10)
        self.assertEqual(learner.kernel_dt, 0.4)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), 0.4)

        learner.kernel_dt = 0.2
        self.assertEqual(learner.kernel_dt, 0.2)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), 0.2)
        self.assertEqual(learner.kernel_size, 20)
        np.testing.assert_array_almost_equal(learner.kernel_discretization, np.linspace(0.0, 4.0, 21))

        learner.kernel_dt = 0.199
        self.assertEqual(learner.kernel_dt, 0.19047619047619047)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), 0.19047619047619047)
        self.assertEqual(learner.kernel_size, 21)

    def test_hawkes_em_get_kernel_values(self):
        em = HawkesEM(kernel_support=4, kernel_size=10, n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)
        self.assertEqual(em.kernel.shape, (self.n_nodes, self.n_nodes, 10))
        self.assertTrue(np.all(em.kernel >= 0.0))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vals = em.get_kernel_values(i, j, em.kernel_discretization[1:])
                self.assertTrue(np.allclose(vals, em.kernel[i, j, :]))

    def test_hawkes_em_kernel_primitives(self):
        em = HawkesEM(kernel_support=4, kernel_size=10, n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events)
        primitives = em._get_kernel_primitives()
        self.assertEqual(primitives.shape, (self.n_nodes, self.n_nodes, 10))
        self.assertTrue(np.all(np.diff(primitives, axis=2) >= 0.0))
        self.assertTrue(np.allclose(em.get_kernel_norms(), primitives[:, :, -1]))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vals = em._compute_primitive_kernel_values(i, j, em.kernel_discretization[1:])
                self.assertTrue(np.allclose(vals, primitives[i, j, :]))

        steps = np.diff(em.kernel_discretization)
        expected = np.cumsum(em.kernel * steps[None, None, :], axis=2)
        self.assertTrue(np.allclose(primitives, expected))
        flat_primitives = em._learner.get_kernel_primitives(em._flat_kernels)
        self.assertTrue(np.allclose(flat_primitives.reshape(primitives.shape), primitives))

    def test_time_changed_interarrival_times_exp_kern(self):
        simu, baseline, adjacency, decays = simulate_small_exp_hawkes()
        kernel_support = compute_approx_support_of_exp_kernel(adjacency, decays, 1e-4)
        kernel_size = 20
        exact_kernel = discretization_of_exp_kernel(2, adjacency, decays, kernel_support, kernel_size)

        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size, tol=1e-5, max_iter=40, verbose=False)
        tcit = em.time_changed_interarrival_times(
            events=[simu.timestamps],
            end_times=simu.end_time,
            baseline=baseline,
            kernel=exact_kernel,
        )
        for node_values in tcit[0]:
            self.assertTrue(np.all(node_values > 0.0))
            self.assertAlmostEqual(float(np.quantile(node_values, 0.5)), 1.0, delta=0.35)

        baseline_start = np.array([0.05 * np.mean(np.diff(ts)) for ts in simu.timestamps])
        np.random.seed(123269)
        kernel_start = np.zeros((2, 2, kernel_size))
        kernel_start[:, :, : kernel_size - 1] = 0.01 * np.cumsum(
            np.random.uniform(size=(2, 2, kernel_size - 1)), axis=2
        )[:, :, ::-1]
        em.fit([simu.timestamps], baseline_start=baseline_start, kernel_start=kernel_start)
        for node_values in em.time_changed_interarrival_times()[0]:
            self.assertTrue(np.all(node_values > 0.0))


if __name__ == "__main__":
    unittest.main()
