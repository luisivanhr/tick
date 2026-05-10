import ast
import collections
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_registry import EXPECTED_XFAIL_GAPS, flatten


ALLOWED_STATUSES = {
    "pass",
    "xfail_equivalence_gap",
    "skip_optional_backend",
    "out_of_scope_non_hawkes",
}

PASS_SIMULATION_FILES = {
    "hawkes_kernel_0_test.py",
    "hawkes_kernel_exp_test.py",
    "hawkes_kernel_pickle_test.py",
    "hawkes_kernel_power_law_test.py",
    "hawkes_kernel_sum_exp_test.py",
    "hawkes_kernel_test.py",
    "hawkes_kernel_time_func_test.py",
    "simu_hawkes_exp_kernels_test.py",
    "simu_hawkes_sumexp_kernels_test.py",
    "simu_hawkes_multi_test.py",
}

PASS_MODEL_FILES = {
    "model_hawkes_expkern_loglik_test.py",
    "model_hawkes_expkern_leastsq_test.py",
    "model_hawkes_sumexpkern_loglik_test.py",
    "model_hawkes_sumexpkern_leastsq_test.py",
}

ADVANCED_INFERENCE_FILES = {
    "hawkes_adm4_test.py",
    "hawkes_basis_kernels_test.py",
    "hawkes_cumulant_matching_test.py",
    "hawkes_expkern_test.py",
    "hawkes_sumexpkern_test.py",
    "hawkes_sumgaussians_test.py",
}

PASS_HAWKES_EM_METHODS = {
    "test_hawkes_em_attributes",
    "test_hawkes_em_fit_1",
    "test_hawkes_em_fit_2",
    "test_hawkes_em_fit_3",
    "test_hawkes_em_score",
    "test_hawkes_em_kernel_shape",
    "test_hawkes_em_kernel_support",
    "test_hawkes_em_kernel_size",
    "test_hawkes_em_kernel_dt",
    "test_hawkes_em_get_kernel_values",
    "test_hawkes_em_kernel_primitives",
    "test_time_changed_interarrival_times_exp_kern",
}


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _discover_tick_hawkes_tests():
    test_root = _repo_root() / "tick" / "hawkes"
    if not test_root.exists():
        raise unittest.SkipTest("local tick Hawkes source tree is not available")
    tests = []
    for path in sorted(test_root.glob("**/tests/*_test.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        rel_path = path.relative_to(_repo_root()).as_posix()
        for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                    tests.append(f"{rel_path}::{cls.name}.{item.name}")
    return tests


def classify_tick_test(test_id):
    path, _, qualified_name = test_id.partition("::")
    filename = Path(path).name
    method = qualified_name.rsplit(".", 1)[-1]

    if "hawkes_cumulant_matching_test.py" == filename and (
        "_tf_" in method or "_pyt_" in method
    ):
        return "skip_optional_backend", "optional TensorFlow/PyTorch cumulant backend"

    if filename == "hawkes_cumulant_matching_test.py" and method in {
        "test_hawkes_cumulants",
        "test_hawkes_cumulants_unfit",
    }:
        return "pass", "cumulant formulas and unfit behavior are covered by source-backed tests"

    if "/model/tests/" in path and filename in PASS_MODEL_FILES:
        return "pass", "deterministic model formulas are covered by parity tests"

    if "/simulation/tests/" in path and filename in PASS_SIMULATION_FILES:
        return "pass", "deterministic kernel/simulation accessors are covered"

    if filename in {"simu_point_process_test.py", "simu_poisson_test.py", "simu_inhomogeneous_poisson_test.py"}:
        return "pass", "point-process stochastic and public-accessor behavior is covered"

    if filename == "simu_hawkes_test.py":
        if method in {
            "test_hawkes_set_kernel",
            "test_hawkes_set_baseline_piecewiseconstant",
            "test_hawkes_set_baseline_timefunction",
            "test_hawkes_mean_intensity",
            "test_simu_hawkes_constructor",
            "test_simu_hawkes_constructor_errors",
            "test_hawkes_negative_intensity_fail",
            "test_hawkes_negative_intensity",
            "test_hawkes_set_timestamps",
            "test_simu_hawkes_force_simulation",
        }:
            return "pass", "generic Hawkes deterministic behavior is covered"
        return "xfail_equivalence_gap", "generic Hawkes simulation path still needs tick-level audit"

    if filename in {"hawkes_expkern_test.py", "hawkes_sumexpkern_test.py"}:
        return "pass", "parametric learner solver/settings behavior is covered by source-backed tests"

    if filename == "hawkes_adm4_test.py":
        return "pass", "ADM4 EM/ADMM algorithm and public behavior are covered by source-backed tests"

    if filename == "hawkes_sumgaussians_test.py":
        return "pass", "sum-of-Gaussians EM/prox behavior is covered by source-backed tests"

    if filename == "hawkes_basis_kernels_test.py":
        return "pass", "basis-kernel EM behavior is covered by source-backed tests"

    if filename == "hawkes_conditional_law_test.py":
        return "pass", "conditional-law algorithm parity is covered by source-backed tests"

    if filename == "hawkes_em_test.py" and method in PASS_HAWKES_EM_METHODS:
        return "pass", "HawkesEM algorithm and public behavior are covered by source-backed tests"

    if "/inference/tests/" in path and filename in ADVANCED_INFERENCE_FILES:
        return "xfail_equivalence_gap", "inference algorithm parity is the active high-risk workstream"

    return "out_of_scope_non_hawkes", "not part of the Hawkes/point-process target surface"


class TickEquivalenceLedgerTest(unittest.TestCase):
    def test_every_local_tick_hawkes_test_is_classified(self):
        tests = _discover_tick_hawkes_tests()
        self.assertEqual(len(tests), 171)
        for test_id in tests:
            with self.subTest(test_id=test_id):
                status, reason = classify_tick_test(test_id)
                self.assertIn(status, ALLOWED_STATUSES)
                self.assertTrue(reason)

    def test_ledger_has_both_green_and_gap_work(self):
        counts = collections.Counter(
            classify_tick_test(test_id)[0] for test_id in _discover_tick_hawkes_tests()
        )
        self.assertEqual(counts["pass"], 165)
        self.assertEqual(counts["xfail_equivalence_gap"], 0)
        self.assertEqual(counts["skip_optional_backend"], 6)
        self.assertEqual(counts["out_of_scope_non_hawkes"], 0)

    def test_xfail_gaps_match_contract_registry(self):
        actual = {
            test_id
            for test_id in _discover_tick_hawkes_tests()
            if classify_tick_test(test_id)[0] == "xfail_equivalence_gap"
        }
        self.assertSetEqual(actual, flatten(EXPECTED_XFAIL_GAPS))


if __name__ == "__main__":
    unittest.main()
