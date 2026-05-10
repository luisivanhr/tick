import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.hawkes.inference import (  # noqa: E402
    HawkesCumulantMatching,
    HawkesTheoreticalCumulant,
)


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _load_train_data():
    fixture_path = (
        _repo_root()
        / "tick"
        / "hawkes"
        / "inference"
        / "tests"
        / "hawkes_cumulant_matching_test-train_data.pkl"
    )
    with fixture_path.open("rb") as fixture:
        return pickle.load(fixture)


def _reference_end_times(realizations):
    out = []
    for realization in realizations:
        out.append(max((float(timestamps[-1]) for timestamps in realization if timestamps.size), default=0.0))
    return np.asarray(out, dtype=float)


def _reference_theoretical_cumulants(baseline, adjacency):
    baseline = np.asarray(baseline, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    R = np.linalg.inv(np.eye(baseline.size, dtype=float) - adjacency)
    mean_intensity = R @ baseline
    covariance = R @ np.diag(mean_intensity) @ R.T
    skewness = np.zeros_like(covariance)
    for i in range(baseline.size):
        for k in range(baseline.size):
            value = 0.0
            for m in range(baseline.size):
                r_im = R[i, m]
                r_km = R[k, m]
                value += (
                    r_im * r_im * covariance[k, m]
                    + 2.0 * r_im * r_km * covariance[i, m]
                    - 2.0 * mean_intensity[m] * r_im * r_im * r_km
                )
            skewness[i, k] = value
    return R, mean_intensity, covariance, skewness


def _reference_A_and_I(timestamps_i, timestamps_j, end_time, support, mean_intensity_j):
    n_j = len(timestamps_j)
    res_C = 0.0
    res_J = 0.0
    width = 2.0 * support
    trend_C_j = mean_intensity_j * width
    trend_J_j = mean_intensity_j * width * width
    last_l = 0
    for t_i_k in timestamps_i:
        if t_i_k - support < 0:
            continue
        while last_l < n_j:
            if timestamps_j[last_l] <= t_i_k - width:
                last_l += 1
            else:
                break
        l = last_l
        timestamps_in_interval = 0
        sub_res = 0.0
        while l < n_j:
            abs_delta = abs(timestamps_j[l] - t_i_k)
            if abs_delta < width:
                sub_res += width - abs_delta
                if abs_delta < support:
                    timestamps_in_interval += 1
            else:
                break
            l += 1
        if l == n_j:
            continue
        res_C += timestamps_in_interval - trend_C_j
        res_J += sub_res - trend_J_j
    return res_C / end_time, res_J / end_time


def _reference_E(timestamps_i, timestamps_j, timestamps_k, end_time, support, L_i, L_j, J_ij):
    n_i = len(timestamps_i)
    n_j = len(timestamps_j)
    res = 0.0
    last_l = 0
    last_m = 0
    trend_i = L_i * 2.0 * support
    trend_j = L_j * 2.0 * support
    for tau in timestamps_k:
        if tau - support < 0:
            continue
        while last_l < n_i:
            if timestamps_i[last_l] <= tau - support:
                last_l += 1
            else:
                break
        l = last_l
        while l < n_i:
            if timestamps_i[l] < tau + support:
                l += 1
            else:
                break
        while last_m < n_j:
            if timestamps_j[last_m] <= tau - support:
                last_m += 1
            else:
                break
        m = last_m
        while m < n_j:
            if timestamps_j[m] < tau + support:
                m += 1
            else:
                break
        if m == n_j or l == n_i:
            continue
        res += (l - last_l - trend_i) * (m - last_m - trend_j) - J_ij
    return res / end_time


def _reference_empirical_cumulants(realizations, support):
    end_times = _reference_end_times(realizations)
    n_realizations = len(realizations)
    n_nodes = len(realizations[0])
    L_day = np.zeros((n_realizations, n_nodes), dtype=float)
    for day, realization in enumerate(realizations):
        for i in range(n_nodes):
            L_day[day, i] = len(realization[i]) / end_times[day]
    L = np.mean(L_day, axis=0)
    C = np.zeros((n_nodes, n_nodes), dtype=float)
    J = np.zeros((n_realizations, n_nodes, n_nodes), dtype=float)
    for day in range(n_realizations):
        C_day = np.zeros((n_nodes, n_nodes), dtype=float)
        J_day = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(n_nodes):
                C_day[i, j], J_day[i, j] = _reference_A_and_I(
                    realizations[day][i],
                    realizations[day][j],
                    end_times[day],
                    support,
                    L_day[day, j],
                )
        C_day[:] = 0.5 * (C_day + C_day.T)
        J_day[:] = 0.5 * (J_day + J_day.T)
        C += C_day / n_realizations
        J[day] = J_day
    E_c = np.zeros((n_nodes, n_nodes, 2), dtype=float)
    for day in range(n_realizations):
        for i in range(n_nodes):
            for j in range(n_nodes):
                E_c[i, j, 0] += _reference_E(
                    realizations[day][i],
                    realizations[day][j],
                    realizations[day][j],
                    end_times[day],
                    support,
                    L_day[day, i],
                    L_day[day, j],
                    J[day, i, j],
                )
                E_c[i, j, 1] += _reference_E(
                    realizations[day][j],
                    realizations[day][j],
                    realizations[day][i],
                    end_times[day],
                    support,
                    L_day[day, j],
                    L_day[day, j],
                    J[day, j, j],
                )
    E_c /= n_realizations
    K_c = (2.0 * E_c[:, :, 0] + E_c[:, :, 1]) / 3.0
    return L, C, K_c


class HawkesCumulantMatchingParityTest(unittest.TestCase):
    def test_theoretical_cumulants_match_tick_cpp_formulas(self):
        fixture = _load_train_data()[2.0]
        baseline = fixture["baseline"]
        adjacency = fixture["adjacency"]
        expected_R, expected_L, expected_C, expected_K = _reference_theoretical_cumulants(
            baseline,
            adjacency,
        )

        cumulants = HawkesTheoreticalCumulant(baseline.size)
        self.assertEqual(cumulants.dimension, baseline.size)
        cumulants.baseline = baseline
        cumulants.adjacency = adjacency
        np.testing.assert_allclose(cumulants.baseline, baseline)
        np.testing.assert_allclose(cumulants.adjacency, adjacency)
        np.testing.assert_allclose(cumulants._R, expected_R)
        np.testing.assert_allclose(
            np.eye(cumulants.dimension) - np.linalg.inv(cumulants._R),
            adjacency,
        )

        cumulants.compute_cumulants()
        np.testing.assert_allclose(cumulants.mean_intensity, expected_L, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cumulants.covariance, expected_C, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cumulants.skewness, expected_K, rtol=1e-13, atol=1e-13)

    def test_empirical_cumulants_match_tick_cpp_loop_reference(self):
        for support, fixture in _load_train_data().items():
            with self.subTest(support=support):
                learner = HawkesCumulantMatching(integration_support=support)
                learner._set_data(fixture["timestamps"])
                self.assertFalse(learner._cumulant_computer.cumulants_ready)

                expected_L, expected_C, expected_K = _reference_empirical_cumulants(
                    fixture["timestamps"],
                    support,
                )
                learner.compute_cumulants()
                self.assertTrue(learner._cumulant_computer.cumulants_ready)
                np.testing.assert_allclose(learner.mean_intensity, expected_L, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(learner.covariance, expected_C, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(learner.skewness, expected_K, rtol=1e-12, atol=1e-12)
                self.assertGreater(learner.approximate_optimal_cs_ratio(), 0.0)

                learner._set_data(fixture["timestamps"])
                self.assertTrue(learner._cumulant_computer.cumulants_ready)

    def test_compute_cumulants_without_data_matches_tick_error(self):
        learner = HawkesCumulantMatching(
            100.0,
            cs_ratio=0.9,
            max_iter=299,
            print_every=30,
            step=1e-2,
            solver="adam",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "^Cannot compute cumulants if no realization has been provided$",
        ):
            learner.compute_cumulants()


if __name__ == "__main__":
    unittest.main()
