import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.base import TimeFunction, normalize_events
from our_hawkes.hawkes import (
    HawkesADM4,
    HawkesCumulantMatching,
    HawkesCumulantMatchingTf,
    HawkesEM,
    HawkesExpKern,
    HawkesKernelExp,
    HawkesKernelPowerLaw,
    HawkesKernelSumExp,
    HawkesSumExpKern,
    ModelHawkesExpKernLogLik,
    SimuHawkesExpKernels,
    SimuHawkesMulti,
    SimuPoissonProcess,
)


class HawkesPublicAPITest(unittest.TestCase):
    def test_hawkes_public_all_exports_are_stable(self):
        import our_hawkes.hawkes as hawkes_api

        tick_hawkes_public_names = {
            "TimeFunction",
            "HawkesKernel",
            "HawkesKernel0",
            "HawkesKernelExp",
            "HawkesKernelPowerLaw",
            "HawkesKernelSumExp",
            "HawkesKernelTimeFunc",
            "SimuPoissonProcess",
            "SimuInhomogeneousPoisson",
            "SimuHawkes",
            "SimuHawkesExpKernels",
            "SimuHawkesSumExpKernels",
            "SimuHawkesMulti",
            "ModelHawkesExpKernLeastSq",
            "ModelHawkesExpKernLogLik",
            "ModelHawkesSumExpKernLeastSq",
            "ModelHawkesSumExpKernLogLik",
            "HawkesExpKern",
            "HawkesSumExpKern",
            "HawkesADM4",
            "HawkesSumGaussians",
            "HawkesEM",
            "HawkesBasisKernels",
            "HawkesConditionalLaw",
            "HawkesCumulantMatching",
            "HawkesCumulantMatchingPyT",
            "HawkesCumulantMatchingTf",
        }

        exported_names = set(hawkes_api.__all__)
        self.assertEqual(set(), tick_hawkes_public_names - exported_names)
        for name in hawkes_api.__all__:
            self.assertTrue(hasattr(hawkes_api, name), name)


class ValidationTest(unittest.TestCase):
    def test_normalize_single_and_multiple_realizations(self):
        events = [np.array([0.1, 0.3]), np.array([0.2])]
        realizations, end_times, n_nodes = normalize_events(events)
        self.assertEqual(len(realizations), 1)
        self.assertEqual(n_nodes, 2)
        self.assertAlmostEqual(end_times[0], 0.3)

        realizations, end_times, n_nodes = normalize_events([events, events], 1.0)
        self.assertEqual(len(realizations), 2)
        self.assertEqual(n_nodes, 2)
        self.assertTrue(np.allclose(end_times, [1.0, 1.0]))

    def test_unsorted_events_raise(self):
        with self.assertRaises(ValueError):
            normalize_events([np.array([0.2, 0.1])])


class TimeFunctionAndKernelTest(unittest.TestCase):
    def test_time_function_interpolation_and_border(self):
        tf = TimeFunction((np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 0.0])))
        self.assertAlmostEqual(tf.value(0.5), 1.0)
        self.assertEqual(tf.value(-1.0), 0.0)
        self.assertEqual(tf.value(3.0), 0.0)
        self.assertAlmostEqual(tf.primitive(2.0), 2.0, places=2)

    def test_kernel_formulas(self):
        exp = HawkesKernelExp(0.5, 2.0)
        self.assertAlmostEqual(exp.get_norm(), 0.5)
        self.assertAlmostEqual(exp.get_value(0.0), 1.0)
        self.assertAlmostEqual(exp.get_primitive_value(10.0), 0.5, places=3)

        sumexp = HawkesKernelSumExp([0.2, 0.3], [1.0, 2.0])
        self.assertAlmostEqual(sumexp.get_norm(), 0.5)
        self.assertEqual(sumexp.n_decays, 2)

        power = HawkesKernelPowerLaw(1.0, 1.0, 2.0, support=10.0)
        self.assertGreater(power.get_norm(), 0.0)


class SimulationTest(unittest.TestCase):
    def test_poisson_seed_reproducibility(self):
        p1 = SimuPoissonProcess([0.5, 0.2], end_time=5.0, seed=123, verbose=False).simulate()
        p2 = SimuPoissonProcess([0.5, 0.2], end_time=5.0, seed=123, verbose=False).simulate()
        for a, b in zip(p1.timestamps, p2.timestamps):
            self.assertTrue(np.allclose(a, b))

    def test_hawkes_simulation_metrics_and_compensator(self):
        simu = SimuHawkesExpKernels(
            adjacency=np.array([[0.2, 0.1], [0.0, 0.15]]),
            decays=1.5,
            baseline=np.array([0.4, 0.3]),
            end_time=4.0,
            seed=7,
            verbose=False,
        )
        simu.track_intensity(0.5)
        simu.simulate()
        simu.store_compensator_values()
        self.assertEqual(simu.n_nodes, 2)
        self.assertLess(simu.spectral_radius(), 1.0)
        self.assertEqual(len(simu.tracked_intensity), 2)
        self.assertEqual(len(simu.tracked_compensator), 2)
        self.assertEqual(len(simu.mean_intensity()), 2)

    def test_hawkes_multi(self):
        base = SimuHawkesExpKernels([[0.1]], 1.0, baseline=[0.2], end_time=2.0, seed=5, verbose=False)
        multi = SimuHawkesMulti(base, 2, n_threads=1).simulate()
        self.assertEqual(len(multi.timestamps), 2)


class ModelAndLearnerTest(unittest.TestCase):
    def setUp(self):
        self.events = [np.array([0.2, 0.8, 1.7]), np.array([0.5, 1.2])]
        self.end_time = 2.0

    def test_loglik_model_loss_grad_hessian_norm(self):
        model = ModelHawkesExpKernLogLik(1.2).fit(self.events, self.end_time)
        coeffs = np.array([0.4, 0.3, 0.1, 0.05, 0.02, 0.1])
        loss = model.loss(coeffs)
        grad = model.grad(coeffs)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(grad.shape, coeffs.shape)
        self.assertGreaterEqual(model.hessian_norm(coeffs, np.ones_like(coeffs)), 0.0)

    def test_exp_learner_fit_score_and_kernel_access(self):
        learner = HawkesExpKern(
            1.2,
            gofit="likelihood",
            penalty="none",
            max_iter=5,
            tol=1e-4,
            verbose=False,
        ).fit(self.events, end_times=self.end_time)
        self.assertEqual(learner.adjacency.shape, (2, 2))
        self.assertTrue(math.isfinite(learner.score()))
        values = learner.get_kernel_values(0, 0, np.linspace(0, 1, 4))
        self.assertEqual(values.shape, (4,))

    def test_parametric_learner_solver_start_history_and_intensity_compat(self):
        learner = HawkesExpKern(
            1.2,
            gofit="likelihood",
            penalty="none",
            solver="svrg",
            max_iter=-1,
            warm_start=True,
        )
        learner.fit(self.events, end_times=self.end_time, start=0.4)
        np.testing.assert_allclose(learner.coeffs, np.full(6, 0.4))
        self.assertEqual(learner.history[-1]["n_iter"], 0)

        learner.fit(self.events, end_times=self.end_time)
        np.testing.assert_allclose(learner.coeffs, np.full(6, 0.4))

        with self.assertRaisesRegex(ValueError, "call `fit` before `score`"):
            HawkesExpKern(1.0).score()

        intensities, times = learner.estimated_intensity(self.events, None, end_time=self.end_time)
        self.assertEqual(len(intensities), 2)
        self.assertEqual(intensities[0].shape, times.shape)

        sumexp = HawkesSumExpKern([1.0, 2.0], solver="l-bfgs-b", penalty="elasticnet", max_iter=1)
        sumexp.fit(self.events, end_times=self.end_time)
        self.assertEqual(sumexp.adjacency.shape, (2, 2, 2))
        self.assertGreaterEqual(len(sumexp.history), 1)
        with self.assertRaisesRegex(ValueError, "``penalty`` must be one of"):
            HawkesSumExpKern([1.0, 2.0], penalty="nuclear")

    def test_em_and_adm4_fit(self):
        em = HawkesEM(kernel_support=1.0, kernel_size=4, max_iter=2).fit(self.events, self.end_time)
        self.assertEqual(em.kernel.shape, (2, 2, 4))
        self.assertTrue(math.isfinite(em.score()))

        adm4 = HawkesADM4(1.0, max_iter=2, C=10.0, verbose=False).fit(self.events, self.end_time)
        self.assertEqual(adm4.adjacency.shape, (2, 2))

    def test_adm4_start_score_objective_and_intensity_compat(self):
        with self.assertRaisesRegex(ValueError, "call `fit` before `score`"):
            HawkesADM4(1.0).score()

        learner = HawkesADM4(
            1.0,
            max_iter=2,
            C=10.0,
            lasso_nuclear_ratio=0.7,
            verbose=False,
            record_every=1,
        )
        learner.fit(
            self.events,
            self.end_time,
            baseline_start=np.zeros(2),
            adjacency_start=np.full((2, 2), 0.2),
        )
        self.assertEqual(learner.coeffs.shape, (6,))
        self.assertTrue(math.isfinite(learner.score()))
        self.assertTrue(math.isfinite(learner.objective(learner.coeffs)))
        self.assertGreaterEqual(len(learner.history), 1)
        self.assertAlmostEqual(learner._prox_l1.strength, 0.07)
        self.assertAlmostEqual(learner._prox_nuclear.strength, 0.03)

        intensities, times = learner.estimated_intensity(self.events, None, end_time=self.end_time)
        self.assertEqual(len(intensities), 2)
        self.assertEqual(intensities[0].shape, times.shape)

    def test_cumulant_matching_and_tf_optional(self):
        cumulant = HawkesCumulantMatching(1.0).fit(self.events, self.end_time)
        self.assertEqual(cumulant.get_kernel_norms().shape, (2, 2))
        with self.assertRaises(ImportError):
            HawkesCumulantMatchingTf(1.0)


if __name__ == "__main__":
    unittest.main()
