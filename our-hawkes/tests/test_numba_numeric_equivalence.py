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

        self.assertGreaterEqual(len(numeric._exp_loglik_loss_scan_numba.signatures), 1)
        self.assertGreaterEqual(len(numeric._sumexp_loglik_loss_scan_numba.signatures), 1)


if __name__ == "__main__":
    unittest.main()
