"""Expected unresolved tick Hawkes equivalence gaps.

The registry is intentionally explicit. When a gap is fixed and the ledger
classifier changes to ``pass``, remove the corresponding test id here.
"""

EXPECTED_XFAIL_GAPS = {}


RESOLVED_IN_THIS_SLICE = {
    "tick/hawkes/simulation/tests/simu_hawkes_test.py": [
        "Test.test_simu_hawkes_force_simulation",
    ],
    "tick/hawkes/simulation/tests/simu_inhomogeneous_poisson_test.py": [
        "Test.test_simulation_1d_inhomogeneous_poisson",
    ],
    "tick/hawkes/simulation/tests/simu_point_process_test.py": [
        "Test.test_simulation_time",
    ],
    "tick/hawkes/simulation/tests/simu_poisson_test.py": [
        "Test.test_simulation_1d_poisson",
        "Test.test_simulation_nd_poisson",
    ],
    "tick/hawkes/inference/tests/hawkes_basis_kernels_test.py": [
        "Test.test_em_basis_kernels",
    ],
    "tick/hawkes/inference/tests/hawkes_sumgaussians_test.py": [
        "Test.test_hawkes_sumgaussians_solution",
        "Test.test_hawkes_sumgaussians_set_data",
        "Test.test_hawkes_sumgaussians_parameters",
        "Test.test_hawkes_sumgaussians_lasso_grouplasso_ratio_parameter",
        "Test.test_hawkes_sumgaussians_C_parameter",
    ],
    "tick/hawkes/inference/tests/hawkes_adm4_test.py": [
        "Test.test_sparse",
        "Test.test_hawkes_adm4_solution",
        "Test.test_hawkes_adm4_score",
        "Test.test_hawkes_adm4_set_data",
        "Test.test_hawkes_adm4_parameters",
    ],
    "tick/hawkes/inference/tests/hawkes_conditional_law_test.py": [
        "Test.test_hawkes_conditional_law_norm",
        "Test.test_hawkes_conditional_law_kernels",
        "Test.test_hawkes_conditional_law_baseline",
        "Test.test_hawkes_conditional_mean_intensity",
        "Test.test_hawkes_quad_method",
        "Test.test_hawkes_claw_method",
        "Test.test_incremental_fit",
    ],
    "tick/hawkes/inference/tests/hawkes_cumulant_matching_test.py": [
        "Test.test_hawkes_cumulants",
        "Test.test_hawkes_cumulants_unfit",
    ],
    "tick/hawkes/inference/tests/hawkes_em_test.py": [
        "Test.test_hawkes_em_attributes",
        "Test.test_hawkes_em_fit_1",
        "Test.test_hawkes_em_fit_2",
        "Test.test_hawkes_em_fit_3",
        "Test.test_hawkes_em_score",
        "Test.test_hawkes_em_kernel_shape",
        "Test.test_hawkes_em_kernel_support",
        "Test.test_hawkes_em_kernel_size",
        "Test.test_hawkes_em_kernel_dt",
        "Test.test_hawkes_em_get_kernel_values",
        "Test.test_hawkes_em_kernel_primitives",
        "Test.test_time_changed_interarrival_times_exp_kern",
    ],
    "tick/hawkes/inference/tests/hawkes_expkern_test.py": [
        "Test.test_HawkesExpKern_fit",
        "Test.test_HawkesExpKern_fit_start",
        "Test.test_HawkesExpKern_score",
        "Test.test_HawkesExpKern_settings",
        "Test.test_HawkesExpKern_model_settings",
        "Test.test_HawkesExpKern_penalty_C",
        "Test.test_HawkesExpKern_penalty_elastic_net_ratio",
        "Test.test_HawkesExpKern_solver_basic_settings",
        "Test.test_HawkesExpKern_solver_step",
        "Test.test_HawkesExpKern_solver_random_state",
        "Test.test_corresponding_simu",
    ],
    "tick/hawkes/inference/tests/hawkes_sumexpkern_test.py": [
        "Test.test_sparse",
        "Test.test_HawkesSumExpKern_fit",
        "Test.test_HawkesSumExpKern_score",
        "Test.test_HawkesSumExpKern_fit_start",
        "Test.test_HawkesSumExpKern_settings",
        "Test.test_HawkesSumExpKern_model_settings",
        "Test.test_HawkesSumExpKern_penalty_C",
        "Test.test_HawkesSumExpKern_penalty_elastic_net_ratio",
        "Test.test_HawkesSumExpKern_solver_basic_settings",
        "Test.test_HawkesSumExpKern_solver_step",
        "Test.test_HawkesSumExpKern_solver_random_state",
        "Test.test_corresponding_simu",
    ],
}


def flatten(registry):
    return {
        f"{path}::{test_name}"
        for path, test_names in registry.items()
        for test_name in test_names
    }
