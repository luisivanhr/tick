import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import our_hawkes.hawkes.numeric as numeric
from our_hawkes.hawkes import (
    HawkesKernelExp,
    HawkesKernelSumExp,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLogLik,
    SimuHawkesExpKernels,
    SimuHawkesSumExpKernels,
    SimuPoissonProcess,
)


class NumbaNumericEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.realization = [
            np.array([0.1, 0.7, 1.4]),
            np.array([0.2, 0.9, 1.6]),
        ]
        self.events, self.sizes = numeric.pack_realization(self.realization)
        self.end_time = 2.0
        self.baseline = np.array([0.35, 0.45])
        self.exp_adjacency = np.array([[0.12, 0.05], [0.08, 0.16]])
        self.exp_decays = np.array([[1.1, 1.7], [0.9, 1.3]])
        self.sumexp_adjacency = np.array(
            [
                [[0.08, 0.03, 0.02], [0.04, 0.02, 0.01]],
                [[0.05, 0.01, 0.03], [0.07, 0.06, 0.02]],
            ]
        )
        self.sumexp_decays = np.array([0.7, 1.4, 2.8])

    def test_exponential_and_sumexponential_convolutions_match_references(self):
        timestamps = self.realization[0]
        time = 1.4

        self.assertAlmostEqual(
            numeric.exp_kernel_convolution(time, timestamps, 0.23, 1.6),
            numeric.exp_kernel_convolution_reference(time, timestamps, 0.23, 1.6),
            places=14,
        )
        self.assertAlmostEqual(
            numeric.exp_kernel_convolution(time, timestamps, 0.23, 1.6, include_current=True),
            numeric.exp_kernel_convolution_reference(
                time, timestamps, 0.23, 1.6, include_current=True
            ),
            places=14,
        )
        self.assertAlmostEqual(
            numeric.sumexp_kernel_convolution(
                time, timestamps, self.sumexp_adjacency[0, 0], self.sumexp_decays
            ),
            numeric.sumexp_kernel_convolution_reference(
                time, timestamps, self.sumexp_adjacency[0, 0], self.sumexp_decays
            ),
            places=14,
        )
        self.assertAlmostEqual(
            numeric.sumexp_kernel_convolution(
                time,
                timestamps,
                self.sumexp_adjacency[0, 0],
                self.sumexp_decays,
                include_current=True,
            ),
            numeric.sumexp_kernel_convolution_reference(
                time,
                timestamps,
                self.sumexp_adjacency[0, 0],
                self.sumexp_decays,
                include_current=True,
            ),
            places=14,
        )

    def test_primitive_convolutions_match_references_and_kernel_methods(self):
        timestamps = self.realization[1]
        time = 1.8
        exp_kernel = HawkesKernelExp(0.31, 1.2)
        sumexp_kernel = HawkesKernelSumExp(self.sumexp_adjacency[1, 1], self.sumexp_decays)

        expected_exp = numeric.exp_kernel_primitive_convolution_reference(
            time, timestamps, exp_kernel.intensity, exp_kernel.decay
        )
        expected_sumexp = numeric.sumexp_kernel_primitive_convolution_reference(
            time, timestamps, sumexp_kernel.intensities, sumexp_kernel.decays
        )

        self.assertAlmostEqual(
            numeric.exp_kernel_primitive_convolution(
                time, timestamps, exp_kernel.intensity, exp_kernel.decay
            ),
            expected_exp,
            places=14,
        )
        self.assertAlmostEqual(exp_kernel.get_primitive_convolution(time, timestamps), expected_exp)
        self.assertAlmostEqual(
            numeric.sumexp_kernel_primitive_convolution(
                time, timestamps, sumexp_kernel.intensities, sumexp_kernel.decays
            ),
            expected_sumexp,
            places=14,
        )
        self.assertAlmostEqual(
            sumexp_kernel.get_primitive_convolution(time, timestamps),
            expected_sumexp,
            places=14,
        )

    def test_intensity_and_compensator_evaluations_match_references(self):
        time = 1.4
        np.testing.assert_allclose(
            numeric.exp_intensity_vector(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            numeric.exp_intensity_vector_reference(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            numeric.sumexp_intensity_vector(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
            ),
            numeric.sumexp_intensity_vector_reference(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
            ),
            rtol=1e-14,
            atol=1e-14,
        )

        for node in range(2):
            self.assertAlmostEqual(
                numeric.exp_compensator_value(
                    node,
                    self.end_time,
                    self.events,
                    self.sizes,
                    self.baseline,
                    self.exp_adjacency,
                    self.exp_decays,
                ),
                numeric.exp_compensator_value_reference(
                    node,
                    self.end_time,
                    self.events,
                    self.sizes,
                    self.baseline,
                    self.exp_adjacency,
                    self.exp_decays,
                ),
                places=14,
            )
            self.assertAlmostEqual(
                numeric.sumexp_compensator_value(
                    node,
                    self.end_time,
                    self.events,
                    self.sizes,
                    self.baseline,
                    self.sumexp_adjacency,
                    self.sumexp_decays,
                ),
                numeric.sumexp_compensator_value_reference(
                    node,
                    self.end_time,
                    self.events,
                    self.sizes,
                    self.baseline,
                    self.sumexp_adjacency,
                    self.sumexp_decays,
                ),
                places=14,
            )

        inhibitory_adjacency = np.array([[0.4, -0.3], [0.2, 0.1]], dtype=float)
        self.assertAlmostEqual(
            numeric.exp_intensity_bound(
                time,
                self.events,
                self.sizes,
                self.baseline,
                inhibitory_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            numeric.exp_intensity_bound_reference(
                time,
                self.events,
                self.sizes,
                self.baseline,
                inhibitory_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            places=14,
        )
        self.assertAlmostEqual(
            numeric.sumexp_intensity_bound(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
                include_current=True,
            ),
            numeric.sumexp_intensity_bound_reference(
                time,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
                include_current=True,
            ),
            places=14,
        )

    def test_homogeneous_poisson_event_helper_matches_reference(self):
        uniforms = np.array(
            [
                [0.10, 0.20],
                [0.35, 0.85],
                [0.40, 0.50],
                [0.80, 0.10],
            ],
            dtype=float,
        )
        intensities = np.array([0.5, 1.0, 0.25], dtype=float)
        ref_times = np.empty(uniforms.shape[0], dtype=float)
        ref_nodes = np.empty(uniforms.shape[0], dtype=np.int64)
        actual_times = np.empty(uniforms.shape[0], dtype=float)
        actual_nodes = np.empty(uniforms.shape[0], dtype=np.int64)

        ref = numeric.homogeneous_poisson_events_reference(
            0.2,
            2.0,
            intensities,
            uniforms,
            ref_times,
            ref_nodes,
        )
        actual = numeric.homogeneous_poisson_events(
            0.2,
            2.0,
            intensities,
            uniforms,
            actual_times,
            actual_nodes,
        )
        self.assertEqual(actual, ref)
        np.testing.assert_allclose(actual_times[: actual[0]], ref_times[: ref[0]], rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual_nodes[: actual[0]], ref_nodes[: ref[0]])

        first = SimuPoissonProcess([0.5, 1.0], end_time=5.0, seed=1234, verbose=False).simulate()
        second = SimuPoissonProcess([0.5, 1.0], end_time=5.0, seed=1234, verbose=False).simulate()
        for left, right in zip(first.timestamps, second.timestamps):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)

    def test_simulation_intensity_and_compensator_routes_match_reference_helpers(self):
        exp_simu = SimuHawkesExpKernels(
            self.exp_adjacency,
            self.exp_decays,
            baseline=self.baseline,
            verbose=False,
        ).set_timestamps(self.realization, self.end_time)
        np.testing.assert_allclose(
            exp_simu._intensity_at(1.4, include_current_jumps=True),
            numeric.exp_intensity_vector_reference(
                1.4,
                self.events,
                self.sizes,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertAlmostEqual(
            exp_simu._total_intensity_bound(1.4, include_current_jumps=True),
            numeric.exp_intensity_bound_reference(
                1.4,
                self.events,
                self.sizes,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
                include_current=True,
            ),
            places=14,
        )
        self.assertAlmostEqual(
            exp_simu._evaluate_compensator(1, self.end_time),
            numeric.exp_compensator_value_reference(
                1,
                self.end_time,
                self.events,
                self.sizes,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
            ),
            places=14,
        )

        sumexp_simu = SimuHawkesSumExpKernels(
            self.sumexp_adjacency,
            self.sumexp_decays,
            baseline=self.baseline,
            verbose=False,
        ).set_timestamps(self.realization, self.end_time)
        np.testing.assert_allclose(
            sumexp_simu._intensity_at(1.4, include_current_jumps=True),
            numeric.sumexp_intensity_vector_reference(
                1.4,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
                include_current=True,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertAlmostEqual(
            sumexp_simu._total_intensity_bound(1.4, include_current_jumps=True),
            numeric.sumexp_intensity_bound_reference(
                1.4,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
                include_current=True,
            ),
            places=14,
        )
        self.assertAlmostEqual(
            sumexp_simu._evaluate_compensator(0, self.end_time),
            numeric.sumexp_compensator_value_reference(
                0,
                self.end_time,
                self.events,
                self.sizes,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
            ),
            places=14,
        )

    def test_loglikelihood_scans_match_references_and_model_losses(self):
        exp_value = numeric.exp_loglik_loss_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
        )
        self.assertAlmostEqual(
            exp_value,
            numeric.exp_loglik_loss_scan_reference(
                self.events,
                self.sizes,
                self.end_time,
                self.baseline,
                self.exp_adjacency,
                self.exp_decays,
            ),
            places=14,
        )

        shared_decay = 1.3
        shared_decays = np.full((2, 2), shared_decay)
        exp_model = ModelHawkesExpKernLogLik(shared_decay).fit(self.realization, self.end_time)
        exp_coeffs = np.hstack((self.baseline, self.exp_adjacency.ravel()))
        expected_model_loss = numeric.exp_loglik_loss_scan_reference(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            shared_decays,
        ) / sum(self.sizes)
        self.assertAlmostEqual(exp_model.loss(exp_coeffs), expected_model_loss, places=14)

        sumexp_value = numeric.sumexp_loglik_loss_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
        )
        self.assertAlmostEqual(
            sumexp_value,
            numeric.sumexp_loglik_loss_scan_reference(
                self.events,
                self.sizes,
                self.end_time,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
            ),
            places=14,
        )

        sumexp_model = ModelHawkesSumExpKernLogLik(self.sumexp_decays).fit(
            self.realization, self.end_time
        )
        sumexp_coeffs = np.hstack((self.baseline, self.sumexp_adjacency.ravel()))
        self.assertAlmostEqual(
            sumexp_model.loss(sumexp_coeffs),
            numeric.sumexp_loglik_loss_scan_reference(
                self.events,
                self.sizes,
                self.end_time,
                self.baseline,
                self.sumexp_adjacency,
                self.sumexp_decays,
            )
            / sum(self.sizes),
            places=14,
        )

    def test_packed_multi_realization_and_event_table_helpers(self):
        realizations = [self.realization, [np.array([], dtype=float), np.array([0.3, 0.4])]]
        events, sizes, end_times = numeric.pack_realizations(realizations, end_times=[2.0, 0.5])
        self.assertEqual(events.shape, (2, 2, 3))
        np.testing.assert_array_equal(sizes, np.array([[3, 3], [0, 2]]))
        np.testing.assert_allclose(end_times, np.array([2.0, 0.5]))

        times, nodes = numeric.pack_event_table(self.realization)
        np.testing.assert_allclose(times, np.array([0.1, 0.2, 0.7, 0.9, 1.4, 1.6]))
        np.testing.assert_array_equal(nodes, np.array([0, 1, 0, 1, 0, 1]))

    def test_loglikelihood_gradients_match_references_and_model_gradients(self):
        exp_grad = numeric.exp_loglik_grad_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
        )
        exp_ref = numeric.exp_loglik_grad_scan_reference(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
        )
        np.testing.assert_allclose(exp_grad[0], exp_ref[0], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(exp_grad[1], exp_ref[1], rtol=1e-14, atol=1e-14)

        shared_decay = 1.3
        shared_decays = np.full((2, 2), shared_decay)
        exp_model = ModelHawkesExpKernLogLik(shared_decay).fit(self.realization, self.end_time)
        exp_coeffs = np.hstack((self.baseline, self.exp_adjacency.ravel()))
        exp_expected = numeric.exp_loglik_grad_scan_reference(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            shared_decays,
        )
        np.testing.assert_allclose(
            exp_model.grad(exp_coeffs),
            np.hstack((exp_expected[0], exp_expected[1].ravel())) / sum(self.sizes),
            rtol=1e-14,
            atol=1e-14,
        )

        sumexp_grad = numeric.sumexp_loglik_grad_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
        )
        sumexp_ref = numeric.sumexp_loglik_grad_scan_reference(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
        )
        np.testing.assert_allclose(sumexp_grad[0], sumexp_ref[0], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(sumexp_grad[1], sumexp_ref[1], rtol=1e-14, atol=1e-14)

        sumexp_model = ModelHawkesSumExpKernLogLik(self.sumexp_decays).fit(
            self.realization, self.end_time
        )
        sumexp_coeffs = np.hstack((self.baseline, self.sumexp_adjacency.ravel()))
        np.testing.assert_allclose(
            sumexp_model.grad(sumexp_coeffs),
            np.hstack((sumexp_ref[0], sumexp_ref[1].ravel())) / sum(self.sizes),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_least_squares_statistics_match_references_on_edge_cases(self):
        edge_realization = [
            np.array([], dtype=float),
            np.array([0.2, 0.2, 1.0], dtype=float),
        ]
        edge_events, edge_sizes = numeric.pack_realization(edge_realization)
        target_decays = np.array([1.1, 1.7])

        exp_stats = numeric.exp_ls_statistics(edge_events, edge_sizes, 1.5, target_decays, 1)
        exp_ref = numeric.exp_ls_statistics_reference(edge_events, edge_sizes, 1.5, target_decays, 1)
        for actual, expected in zip(exp_stats, exp_ref):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

        sum_stats = numeric.sumexp_ls_integral_statistics(
            edge_events, edge_sizes, 1.5, self.sumexp_decays
        )
        sum_ref = numeric.sumexp_ls_integral_statistics_reference(
            edge_events, edge_sizes, 1.5, self.sumexp_decays
        )
        for actual, expected in zip(sum_stats, sum_ref):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

        np.testing.assert_allclose(
            numeric.sumexp_ls_event_feature_sums(edge_events, edge_sizes, self.sumexp_decays, 1),
            numeric.sumexp_ls_event_feature_sums_reference(edge_events, edge_sizes, self.sumexp_decays, 1),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_numba_can_be_disabled_by_environment(self):
        env = os.environ.copy()
        env["OUR_HAWKES_DISABLE_NUMBA"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        command = [
            sys.executable,
            "-c",
            "from our_hawkes.hawkes import numeric; "
            "print(numeric.NUMBA_AVAILABLE); "
            "assert not numeric.NUMBA_AVAILABLE; "
            "assert not numeric.is_numba_enabled()",
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        self.assertIn("False", completed.stdout)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_numba_dispatchers_compile_for_equivalence_cases(self):
        numeric.exp_loglik_loss_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
        )
        numeric.sumexp_loglik_loss_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
        )
        numeric.exp_loglik_grad_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
        )
        numeric.sumexp_loglik_grad_scan(
            self.events,
            self.sizes,
            self.end_time,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
        )
        numeric.exp_ls_statistics(self.events, self.sizes, self.end_time, self.exp_decays[0], 0)
        numeric.exp_intensity_bound(
            1.4,
            self.events,
            self.sizes,
            self.baseline,
            self.exp_adjacency,
            self.exp_decays,
            include_current=True,
        )
        numeric.sumexp_intensity_bound(
            1.4,
            self.events,
            self.sizes,
            self.baseline,
            self.sumexp_adjacency,
            self.sumexp_decays,
            include_current=True,
        )
        out_times = np.empty(4, dtype=float)
        out_nodes = np.empty(4, dtype=np.int64)
        numeric.homogeneous_poisson_events(
            0.0,
            2.0,
            np.array([0.5, 1.0]),
            np.array([[0.1, 0.2], [0.4, 0.8], [0.2, 0.6], [0.9, 0.1]]),
            out_times,
            out_nodes,
        )

        self.assertGreaterEqual(len(numeric._exp_loglik_loss_scan_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._sumexp_loglik_loss_scan_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._exp_loglik_grad_scan_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._sumexp_loglik_grad_scan_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._exp_ls_statistics_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._exp_intensity_bound_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._sumexp_intensity_bound_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._homogeneous_poisson_events_numba.signatures), 1)


if __name__ == "__main__":
    unittest.main()
