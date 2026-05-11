import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import numeric  # noqa: E402
from our_hawkes.hawkes import inference  # noqa: E402


class NumbaLearnerEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.realization = [
            np.array([0.1, 0.7, 1.4], dtype=float),
            np.array([0.2, 0.9], dtype=float),
        ]
        self.end_time = 2.0
        self.kernel_dt = 0.5
        self.basis_kernels = np.array(
            [
                [0.4, 0.2, 0.1],
                [0.3, 0.5, 0.25],
            ],
            dtype=float,
        )
        self.basis_primitives = np.cumsum(self.basis_kernels, axis=1) * self.kernel_dt
        self.amplitude_sums = np.array([0.7, 1.1], dtype=float)
        self.amplitudes_u = np.array(
            [
                [0.6, 0.2],
                [0.1, 0.8],
            ],
            dtype=float,
        )

    def test_basis_helper_wrappers_match_references(self):
        r_ref = np.zeros(2, dtype=float)
        r_actual = np.zeros(2, dtype=float)
        inference._basis_compute_r_reference(
            self.realization[0],
            self.end_time,
            self.kernel_dt,
            self.basis_kernels,
            self.basis_primitives,
            r_ref,
        )
        inference._basis_compute_r(
            self.realization[0],
            self.end_time,
            self.kernel_dt,
            self.basis_kernels,
            self.basis_primitives,
            r_actual,
        )
        np.testing.assert_allclose(r_actual, r_ref, rtol=1e-14, atol=1e-14)

        c_ref = np.zeros_like(self.basis_kernels)
        c_actual = np.zeros_like(self.basis_kernels)
        inference._basis_compute_C_reference(
            self.realization[1],
            self.end_time,
            self.kernel_dt,
            self.basis_kernels,
            self.amplitude_sums,
            c_ref,
        )
        inference._basis_compute_C(
            self.realization[1],
            self.end_time,
            self.kernel_dt,
            self.basis_kernels,
            self.amplitude_sums,
            c_actual,
        )
        np.testing.assert_allclose(c_actual, c_ref, rtol=1e-14, atol=1e-14)

        qvd_ref = np.zeros((2, 2), dtype=float)
        qvd_actual = np.zeros((2, 2), dtype=float)
        ddm_ref = np.zeros_like(self.basis_kernels)
        ddm_actual = np.zeros_like(self.basis_kernels)
        mu_ref = inference._basis_compute_mu_q_D_reference(
            0,
            self.realization,
            self.kernel_dt,
            self.basis_kernels,
            self.amplitudes_u,
            0.7,
            qvd_ref,
            ddm_ref,
        )
        mu_actual = inference._basis_compute_mu_q_D(
            0,
            self.realization,
            self.kernel_dt,
            self.basis_kernels,
            self.amplitudes_u,
            0.7,
            qvd_actual,
            ddm_actual,
        )
        self.assertAlmostEqual(mu_actual, mu_ref, places=14)
        np.testing.assert_allclose(qvd_actual, qvd_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(ddm_actual, ddm_ref, rtol=1e-14, atol=1e-14)

        kernel_ref = np.empty(3, dtype=float)
        kernel_actual = np.empty(3, dtype=float)
        cdm = np.array([0.2, 0.4, 0.1], dtype=float)
        ddm = np.array([0.3, 0.2, 0.05], dtype=float)
        err_ref = inference._basis_compute_gdm_reference(1.2, self.kernel_dt, kernel_ref, cdm, ddm, 1e-8, 20)
        err_actual = inference._basis_compute_gdm(1.2, self.kernel_dt, kernel_actual, cdm, ddm, 1e-8, 20)
        self.assertAlmostEqual(err_actual, err_ref, places=14)
        np.testing.assert_allclose(kernel_actual, kernel_ref, rtol=1e-14, atol=1e-14)

    def test_sumgaussians_weight_and_em_helpers_match_references(self):
        realizations = [
            self.realization,
            [
                np.array([0.15, 0.55], dtype=float),
                np.array([0.25, 0.95, 1.5], dtype=float),
            ],
        ]
        end_times = np.array([2.0, 1.8], dtype=float)
        events, sizes, end_times = numeric.pack_realizations(realizations, end_times)
        means = np.array([0.0, 1.25, 2.5], dtype=float)
        std = 2.5 / (3.0 * np.pi)

        g_ref, map_ref = inference._sumgaussians_compute_weights_reference(
            events,
            sizes,
            end_times,
            means,
            std,
        )
        g_actual, map_actual = inference._sumgaussians_compute_weights(
            events,
            sizes,
            end_times,
            means,
            std,
        )
        np.testing.assert_allclose(g_actual, g_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(map_actual, map_ref, rtol=1e-14, atol=1e-14)

        kernel_integral = np.sum(map_ref, axis=0)
        mu_ref = np.array([0.25, 0.35], dtype=float)
        mu_actual = mu_ref.copy()
        amplitudes_ref = np.array(
            [
                [0.30, 0.20, 0.10, 0.25, 0.15, 0.05],
                [0.18, 0.22, 0.12, 0.28, 0.16, 0.08],
            ],
            dtype=float,
        )
        amplitudes_actual = amplitudes_ref.copy()
        next_mu_ref, next_c_ref = inference._sumgaussians_em_inner_loop_reference(
            g_ref,
            sizes,
            end_times,
            kernel_integral,
            0.07,
            0.03,
            2,
            mu_ref,
            amplitudes_ref,
        )
        next_mu_actual, next_c_actual = inference._sumgaussians_em_inner_loop(
            g_actual,
            sizes,
            end_times,
            kernel_integral,
            0.07,
            0.03,
            2,
            mu_actual,
            amplitudes_actual,
        )
        np.testing.assert_allclose(mu_actual, mu_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(amplitudes_actual, amplitudes_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(next_mu_actual, next_mu_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(next_c_actual, next_c_ref, rtol=1e-14, atol=1e-14)

    def test_adm4_weight_and_em_helpers_match_references(self):
        realizations = [
            self.realization,
            [
                np.array([0.15, 0.55], dtype=float),
                np.array([0.25, 0.95, 1.5], dtype=float),
            ],
        ]
        end_times = np.array([2.0, 1.8], dtype=float)
        events, sizes, end_times = numeric.pack_realizations(realizations, end_times)
        decay = 1.3

        g_ref, map_ref = inference._adm4_compute_weights_reference(events, sizes, end_times, decay)
        g_actual, map_actual = inference._adm4_compute_weights(events, sizes, end_times, decay)
        np.testing.assert_allclose(g_actual, g_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(map_actual, map_ref, rtol=1e-14, atol=1e-14)

        kernel_integral = np.sum(map_ref, axis=0)
        mu_ref = np.array([0.25, 0.35], dtype=float)
        mu_actual = mu_ref.copy()
        adjacency_ref = np.array([[0.30, 0.10], [0.20, 0.40]], dtype=float)
        adjacency_actual = adjacency_ref.copy()
        z1 = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=float)
        z2 = np.array([[0.02, 0.01], [0.04, 0.03]], dtype=float)
        u1 = np.array([[0.005, 0.002], [0.001, 0.004]], dtype=float)
        u2 = np.array([[0.003, 0.006], [0.002, 0.005]], dtype=float)

        next_mu_ref, next_c_ref = inference._adm4_em_update_reference(
            g_ref,
            sizes,
            end_times,
            kernel_integral,
            0.7,
            mu_ref,
            adjacency_ref,
            z1,
            z2,
            u1,
            u2,
        )
        next_mu_actual, next_c_actual = inference._adm4_em_update(
            g_actual,
            sizes,
            end_times,
            kernel_integral,
            0.7,
            mu_actual,
            adjacency_actual,
            z1,
            z2,
            u1,
            u2,
        )
        np.testing.assert_allclose(mu_actual, mu_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(adjacency_actual, adjacency_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(next_mu_actual, next_mu_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(next_c_actual, next_c_ref, rtol=1e-14, atol=1e-14)

    def test_hawkes_em_step_matches_reference_on_edge_cases(self):
        realizations = [
            [
                np.array([], dtype=float),
                np.array([0.2, 0.2, 0.7], dtype=float),
                np.array([0.1, 0.4], dtype=float),
            ],
            [
                np.array([0.05, 0.4], dtype=float),
                np.array([], dtype=float),
                np.array([0.4, 0.9], dtype=float),
            ],
        ]
        end_times = np.array([1.2, 1.1], dtype=float)
        events, sizes, end_times = numeric.pack_realizations(realizations, end_times)
        discretization = np.array([0.0, 0.25, 0.75, 1.5], dtype=float)
        baseline = np.array([0.3, 0.4, 0.2], dtype=float)
        kernel = np.arange(1, 28, dtype=float).reshape(3, 3, 3) / 50.0

        baseline_ref, kernel_ref = inference._hawkes_em_step_reference(
            events,
            sizes,
            end_times,
            baseline,
            kernel,
            discretization,
        )
        baseline_actual, kernel_actual = inference._hawkes_em_step(
            events,
            sizes,
            end_times,
            baseline,
            kernel,
            discretization,
        )
        np.testing.assert_allclose(baseline_actual, baseline_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(kernel_actual, kernel_ref, rtol=1e-14, atol=1e-14)

        baseline_start = baseline.copy()
        kernel_start = kernel.copy()
        learner = inference.HawkesEM(kernel_discretization=discretization, max_iter=1, verbose=False).fit(
            realizations,
            end_times=end_times,
            baseline_start=baseline_start,
            kernel_start=kernel_start,
        )
        baseline_start[0] = 99.0
        kernel_start[0, 0, 0] = 99.0
        np.testing.assert_allclose(learner.baseline, baseline_ref, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(learner.kernel, kernel_ref, rtol=1e-14, atol=1e-14)

    def test_cumulant_counting_helpers_match_references_on_edge_cases(self):
        realizations = [
            [
                np.array([], dtype=float),
                np.array([0.2, 0.2, 0.7], dtype=float),
                np.array([0.1, 0.4], dtype=float),
            ],
            [
                np.array([0.05, 0.4], dtype=float),
                np.array([], dtype=float),
                np.array([0.4, 0.9], dtype=float),
            ],
        ]
        end_times = np.array([1.2, 1.1], dtype=float)
        support = 0.25
        packed_events, packed_sizes, packed_end_times = numeric.pack_realizations(realizations, end_times)
        n_realizations, n_nodes, _ = packed_events.shape
        L_day = packed_sizes / packed_end_times[:, None]
        C_expected = np.zeros((n_nodes, n_nodes), dtype=float)
        J_expected = np.zeros((n_realizations, n_nodes, n_nodes), dtype=float)

        for r in range(n_realizations):
            C_day_ref = np.zeros((n_nodes, n_nodes), dtype=float)
            C_day_actual = np.zeros((n_nodes, n_nodes), dtype=float)
            J_day_ref = np.zeros((n_nodes, n_nodes), dtype=float)
            J_day_actual = np.zeros((n_nodes, n_nodes), dtype=float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    C_day_ref[i, j], J_day_ref[i, j] = inference._cumulant_compute_A_and_I_reference(
                        packed_events[r],
                        packed_sizes[r],
                        i,
                        j,
                        packed_end_times[r],
                        support,
                        L_day[r, j],
                    )
                    C_day_actual[i, j], J_day_actual[i, j] = inference._cumulant_compute_A_and_I(
                        packed_events[r],
                        packed_sizes[r],
                        i,
                        j,
                        packed_end_times[r],
                        support,
                        L_day[r, j],
                    )
            np.testing.assert_allclose(C_day_actual, C_day_ref, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(J_day_actual, J_day_ref, rtol=1e-14, atol=1e-14)
            C_day_ref[:] = 0.5 * (C_day_ref + C_day_ref.T)
            J_day_ref[:] = 0.5 * (J_day_ref + J_day_ref.T)
            C_expected += C_day_ref / n_realizations
            J_expected[r] = J_day_ref

            for i, j, k in [(0, 1, 2), (2, 0, 0), (1, 2, 0), (2, 2, 1)]:
                e_ref = inference._cumulant_compute_E_reference(
                    packed_events[r],
                    packed_sizes[r],
                    i,
                    j,
                    k,
                    packed_end_times[r],
                    support,
                    L_day[r, i],
                    L_day[r, j],
                    J_day_ref[i, j],
                )
                e_actual = inference._cumulant_compute_E(
                    packed_events[r],
                    packed_sizes[r],
                    i,
                    j,
                    k,
                    packed_end_times[r],
                    support,
                    L_day[r, i],
                    L_day[r, j],
                    J_day_ref[i, j],
                )
                self.assertAlmostEqual(e_actual, e_ref, places=14)

        E_expected = np.zeros((n_nodes, n_nodes, 2), dtype=float)
        for r in range(n_realizations):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    E_expected[i, j, 0] += inference._cumulant_compute_E_reference(
                        packed_events[r],
                        packed_sizes[r],
                        i,
                        j,
                        j,
                        packed_end_times[r],
                        support,
                        L_day[r, i],
                        L_day[r, j],
                        J_expected[r, i, j],
                    )
                    E_expected[i, j, 1] += inference._cumulant_compute_E_reference(
                        packed_events[r],
                        packed_sizes[r],
                        j,
                        j,
                        i,
                        packed_end_times[r],
                        support,
                        L_day[r, j],
                        L_day[r, j],
                        J_expected[r, j, j],
                    )
        K_expected = (2.0 * E_expected[:, :, 0] + E_expected[:, :, 1]) / (3.0 * n_realizations)

        learner = inference.HawkesCumulantMatching(support)
        learner._set_data(realizations, end_times=end_times)
        learner.compute_cumulants()
        np.testing.assert_allclose(learner.mean_intensity, np.mean(L_day, axis=0), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(learner.covariance, C_expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(learner.skewness, K_expected, rtol=1e-14, atol=1e-14)

    def test_conditional_law_counting_helper_matches_reference_on_marked_data(self):
        realization = [
            np.array([0.1, 0.2, 0.55, 0.9, 1.4], dtype=float),
            np.array([0.05, 0.2, 0.4, 0.8], dtype=float),
        ]
        marks = [
            np.array([0.5, 1.5, 2.2, 4.0, 6.0], dtype=float),
            np.array([0.4, 1.1, 2.6, 5.7], dtype=float),
        ]
        events, sizes, packed_marks = inference._conditional_law_pack_realization(realization, marks)
        lags = np.array([0.0, 0.2, 0.5, 1.0], dtype=float)

        for y_node, z_node, zmin, zmax in [(0, 1, 0.5, 2.0), (1, 0, -np.inf, np.inf)]:
            y_lambda = sizes[y_node] / 1.6
            x_ref, y_ref = inference._conditional_law_point_process_cond_law_reference(
                events,
                sizes,
                packed_marks,
                y_node,
                z_node,
                lags,
                zmin,
                zmax,
                1.6,
                y_lambda,
            )
            x_actual, y_actual = inference._conditional_law_point_process_cond_law(
                events,
                sizes,
                packed_marks,
                y_node,
                z_node,
                lags,
                zmin,
                zmax,
                1.6,
                y_lambda,
            )
            np.testing.assert_allclose(x_actual, x_ref, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(y_actual, y_ref, rtol=1e-14, atol=1e-14)

    def test_conditional_law_M_V_helpers_match_references(self):
        events = [
            (
                np.array([0.1, 0.3, 0.8, 1.4], dtype=float),
                np.array([0.5, 1.5, 3.0, 5.0], dtype=float),
            ),
            (
                np.array([0.2, 0.9, 1.1], dtype=float),
                np.array([0.2, 1.4, 2.9], dtype=float),
            ),
        ]

        for method in ("gauss", "gauss-", "lin"):
            with self.subTest(method=method):
                learner = inference.HawkesConditionalLaw(
                    marked_components={0: [1.0], 1: [1.0]},
                    n_quad=4,
                    max_support=1.2,
                    max_lag=2.0,
                    delta_lag=0.2,
                    quad_method=method,
                ).fit(events, T=1.6)
                (
                    index2ijl,
                    ijl_to_index,
                    mark_probabilities,
                    int_x,
                    int_y,
                    int_sizes,
                    ig_x,
                    ig_y,
                    ig_sizes,
                    ig2_x,
                    ig2_y,
                    ig2_sizes,
                ) = learner._conditional_linear_pack

                index_first = 0
                for i in range(learner.n_nodes):
                    index_last = index_first
                    for index_last in range(index_first, learner._n_index):
                        i1, _, _ = learner._index2ijl[index_last]
                        if i1 != i:
                            index_last -= 1
                            break
                    n_index = index_last - index_first + 1
                    V_ref = inference._conditional_law_compute_V_reference(
                        i,
                        n_index,
                        learner.n_quad,
                        index_first,
                        index_last,
                        index2ijl,
                        ijl_to_index,
                        learner._quad_x,
                        int_x,
                        int_y,
                        int_sizes,
                    )
                    V_actual = inference._conditional_law_compute_V(
                        i,
                        n_index,
                        learner.n_quad,
                        index_first,
                        index_last,
                        index2ijl,
                        ijl_to_index,
                        learner._quad_x,
                        int_x,
                        int_y,
                        int_sizes,
                    )
                    M_ref = inference._conditional_law_compute_M_reference(
                        n_index,
                        learner.n_quad,
                        index_first,
                        index_last,
                        inference._conditional_law_method_code(method),
                        index2ijl,
                        ijl_to_index,
                        learner.mean_intensity,
                        mark_probabilities,
                        learner._quad_x,
                        learner._quad_w,
                        int_x,
                        int_y,
                        int_sizes,
                        ig_x,
                        ig_y,
                        ig_sizes,
                        ig2_x,
                        ig2_y,
                        ig2_sizes,
                    )
                    M_actual = inference._conditional_law_compute_M(
                        n_index,
                        learner.n_quad,
                        index_first,
                        index_last,
                        method,
                        index2ijl,
                        ijl_to_index,
                        learner.mean_intensity,
                        mark_probabilities,
                        learner._quad_x,
                        learner._quad_w,
                        int_x,
                        int_y,
                        int_sizes,
                        ig_x,
                        ig_y,
                        ig_sizes,
                        ig2_x,
                        ig2_y,
                        ig2_sizes,
                    )
                    np.testing.assert_allclose(V_actual, V_ref, rtol=1e-13, atol=1e-13)
                    np.testing.assert_allclose(M_actual, M_ref, rtol=1e-13, atol=1e-13)
                    index_first = index_last + 1

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_basis_dispatchers_compile(self):
        self.test_basis_helper_wrappers_match_references()
        self.assertGreaterEqual(len(inference._basis_compute_r_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._basis_compute_C_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._basis_compute_mu_q_D_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._basis_compute_gdm_numba.signatures), 1)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_sumgaussians_dispatchers_compile(self):
        self.test_sumgaussians_weight_and_em_helpers_match_references()
        self.assertGreaterEqual(len(inference._sumgaussians_compute_weights_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._sumgaussians_em_inner_loop_numba.signatures), 1)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_adm4_dispatchers_compile(self):
        self.test_adm4_weight_and_em_helpers_match_references()
        self.assertGreaterEqual(len(inference._adm4_compute_weights_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._adm4_em_update_numba.signatures), 1)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_hawkes_em_dispatcher_compiles(self):
        self.test_hawkes_em_step_matches_reference_on_edge_cases()
        self.assertGreaterEqual(len(inference._hawkes_em_step_numba.signatures), 1)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_cumulant_counting_dispatchers_compile(self):
        self.test_cumulant_counting_helpers_match_references_on_edge_cases()
        self.assertGreaterEqual(len(inference._cumulant_compute_A_and_I_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._cumulant_compute_E_numba.signatures), 1)

    @unittest.skipUnless(numeric.NUMBA_AVAILABLE, "Numba is not installed")
    def test_conditional_law_dispatchers_compile(self):
        self.test_conditional_law_counting_helper_matches_reference_on_marked_data()
        self.test_conditional_law_M_V_helpers_match_references()
        self.assertGreaterEqual(len(inference._conditional_law_point_process_cond_law_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._conditional_law_compute_V_numba.signatures), 1)
        self.assertGreaterEqual(len(inference._conditional_law_compute_M_numba.signatures), 1)


if __name__ == "__main__":
    unittest.main()
