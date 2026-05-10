"""Source-backed parity checks for tick parametric Hawkes learners.

The assertions in this file are ported from:
- tick/hawkes/inference/tests/hawkes_expkern_test.py
- tick/hawkes/inference/tests/hawkes_sumexpkern_test.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.hawkes import (  # noqa: E402
    HawkesExpKern,
    HawkesSumExpKern,
    ModelHawkesExpKernLeastSq,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLeastSq,
    SimuHawkesExpKernels,
    SimuHawkesSumExpKernels,
)
from our_hawkes.hawkes.solvers import (  # noqa: E402
    AGD,
    BFGS,
    GD,
    ProxElasticNet,
    ProxL1,
    ProxL2Sq,
    ProxNuclear,
    ProxPositive,
    SGD,
    SVRG,
)


SOLVERS = ["gd", "agd", "svrg", "bfgs", "sgd"]
EXP_PENALTIES = ["none", "l2", "l1", "nuclear", "elasticnet"]
SUMEXP_PENALTIES = ["none", "l2", "l1", "elasticnet"]
GOFITS = ["least-squares", "likelihood"]


class ParametricLearnerParityMixin:
    learner_cls = None
    penalties = None

    def assert_common_solver_settings(self, make_learner):
        solver_class_map = {"gd": GD, "agd": AGD, "sgd": SGD, "svrg": SVRG, "bfgs": BFGS}
        for solver in SOLVERS:
            learner = make_learner(solver=solver)
            self.assertIsInstance(learner._solver_obj, solver_class_map[solver])
            with self.assertRaisesRegex(AttributeError, f"solver is readonly in {learner.__class__.__name__}"):
                learner.solver = solver

            learner = make_learner(solver=solver, tol=self.float_1)
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            learner = make_learner(solver=solver, max_iter=self.int_1)
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            learner = make_learner(solver=solver, verbose=True)
            self.assertTrue(learner.verbose)
            self.assertTrue(learner._solver_obj.verbose)
            learner.verbose = False
            self.assertFalse(learner._solver_obj.verbose)

            learner = make_learner(solver=solver, print_every=self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            learner = make_learner(solver=solver, record_every=self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def assert_common_penalty_settings(self, make_learner):
        prox_class_map = {
            "none": ProxPositive,
            "l1": ProxL1,
            "l2": ProxL2Sq,
            "elasticnet": ProxElasticNet,
            "nuclear": ProxNuclear,
        }
        for penalty in self.penalties:
            learner = make_learner(penalty=penalty)
            self.assertIsInstance(learner._prox_obj, prox_class_map[penalty])
            with self.assertRaisesRegex(AttributeError, f"penalty is readonly in {learner.__class__.__name__}"):
                learner.penalty = penalty

            if penalty != "none":
                learner = make_learner(penalty=penalty, C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1.0 / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1.0 / self.float_2)
                with self.assertRaisesRegex(ValueError, r"^``C`` must be positive, got -1$"):
                    make_learner(penalty=penalty, C=-1)
            else:
                learner = make_learner(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, r'^You cannot set C for penalty "none"$'):
                    learner.C = self.float_1

            with self.assertRaisesRegex(ValueError, r"^``C`` must be positive, got -2$"):
                learner.C = -2

    def assert_common_elastic_net_settings(self, make_learner):
        for penalty in self.penalties:
            if penalty == "elasticnet":
                learner = make_learner(
                    penalty=penalty,
                    C=self.float_1,
                    elastic_net_ratio=0.6,
                )
                self.assertEqual(learner.elastic_net_ratio, 0.6)
                self.assertEqual(learner._prox_obj.ratio, 0.6)
                learner.elastic_net_ratio = 0.3
                self.assertEqual(learner._prox_obj.ratio, 0.3)
            else:
                learner = make_learner(penalty=penalty)
                with self.assertWarnsRegex(
                    RuntimeWarning,
                    rf'^Penalty "{penalty}" has no elastic_net_ratio attribute$',
                ):
                    learner.elastic_net_ratio = 0.6

    def assert_common_solver_step_and_random_state(self, make_learner):
        for solver in SOLVERS:
            if solver == "bfgs":
                with self.assertWarnsRegex(RuntimeWarning, r'^Solver "bfgs" has no settable step$'):
                    learner = make_learner(solver=solver, step=1.0)
                self.assertIsNone(learner.step)
            else:
                learner = make_learner(solver=solver, step=self.float_1)
                self.assertEqual(learner.step, self.float_1)
                self.assertEqual(learner._solver_obj.step, self.float_1)
                learner.step = self.float_2
                self.assertEqual(learner._solver_obj.step, self.float_2)

            if solver == "sgd":
                with self.assertWarnsRegex(RuntimeWarning, r"^SGD step needs to be tuned manually$"):
                    make_learner(solver="sgd", max_iter=1).fit(self.events, 0.3)

            if solver in {"bfgs", "agd", "gd"}:
                with self.assertWarnsRegex(
                    RuntimeWarning,
                    rf'^Solver "{solver}" has no settable random_state$',
                ):
                    learner = make_learner(solver=solver, random_state=1)
                self.assertIsNone(learner.random_state)
            else:
                learner = make_learner(solver=solver, random_state=self.int_1)
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)
                with self.assertRaisesRegex(ValueError, r"^random_state must be positive, got -1$"):
                    make_learner(solver=solver, random_state=-1)

            with self.assertRaisesRegex(AttributeError, f"random_state is readonly in {learner.__class__.__name__}"):
                learner.random_state = self.int_2


class HawkesExpKernParityTest(ParametricLearnerParityMixin, unittest.TestCase):
    penalties = EXP_PENALTIES

    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230
        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]
        self.decays = 3.0

    def make_learner(self, **kwargs):
        return HawkesExpKern(self.decays, **kwargs)

    def test_fit_start_matches_tick_start_contract(self):
        n_nodes = len(self.events)
        n_coeffs = n_nodes + n_nodes * n_nodes
        learner = HawkesExpKern(self.decays, max_iter=-1)
        learner.fit(self.events)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs))
        learner.fit(self.events, start=self.float_1)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs) * self.float_1)
        learner.fit(self.events, start=self.int_1)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs) * self.int_1)
        random_coeffs = np.random.rand(n_coeffs)
        learner.fit(self.events, start=random_coeffs)
        np.testing.assert_array_equal(learner.coeffs, random_coeffs)

    def test_score_matches_tick_seeded_reference(self):
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
        learner = HawkesExpKern(self.decays, record_every=1)
        with self.assertRaisesRegex(ValueError, r"^You must either call `fit` before `score` or provide events$"):
            learner.score()
        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes)
        learner.fit(train_events)
        self.assertAlmostEqual(learner.score(), 2.0855840, delta=5e-4)
        self.assertAlmostEqual(
            learner.score(baseline=given_baseline, adjacency=given_adjacency),
            0.59502417,
            delta=1e-8,
        )
        self.assertAlmostEqual(learner.score(test_events), 1.6001762, delta=5e-4)
        self.assertAlmostEqual(
            learner.score(test_events, baseline=given_baseline, adjacency=given_adjacency),
            0.89322199,
            delta=1e-8,
        )

    def test_fit_runs_representative_solver_penalty_gofit_paths(self):
        for gofit in GOFITS:
            for solver, penalty in [("gd", "none"), ("agd", "l2"), ("svrg", "l1"), ("bfgs", "l2"), ("sgd", "elasticnet")]:
                kwargs = dict(gofit=gofit, solver=solver, penalty=penalty, max_iter=2, verbose=False)
                if penalty != "none":
                    kwargs["C"] = 50
                if solver in {"sgd", "svrg"}:
                    kwargs["random_state"] = 179312
                    kwargs["step"] = 1e-3 if solver == "svrg" else 1e-3
                learner = HawkesExpKern(self.decays, **kwargs).fit(self.events, start=0.3)
                self.assertEqual(learner.adjacency.shape, (3, 3))
                self.assertTrue(np.all(np.isfinite(learner.coeffs)))

    def test_settings_model_penalty_solver_and_corresponding_simu(self):
        self.assert_common_solver_settings(self.make_learner)
        self.assert_common_penalty_settings(self.make_learner)
        self.assert_common_elastic_net_settings(self.make_learner)
        self.assert_common_solver_step_and_random_state(self.make_learner)

        with self.assertRaisesRegex(ValueError, r"^``solver`` must be one of agd, bfgs, gd, sgd, svrg, got wrong_name$"):
            HawkesExpKern(self.decays, solver="wrong_name")
        with self.assertRaisesRegex(ValueError, r"^``penalty`` must be one of elasticnet, l1, l2, none, nuclear, got wrong_name$"):
            HawkesExpKern(self.decays, penalty="wrong_name")
        with self.assertRaisesRegex(ValueError, r"^Parameter gofit \(goodness of fit\) must be either 'least-squares' or 'likelihood'$"):
            HawkesExpKern(self.decays, gofit="wrong_name")

        for gofit, model_cls in {
            "least-squares": ModelHawkesExpKernLeastSq,
            "likelihood": ModelHawkesExpKernLogLik,
        }.items():
            learner = HawkesExpKern(self.float_1, gofit=gofit)
            self.assertEqual(learner.decays, self.float_1)
            self.assertIsInstance(learner._model_obj, model_cls)
            self.assertEqual(learner._model_obj.decays, self.float_1)
            with self.assertRaisesRegex(AttributeError, "decays is readonly in HawkesExpKern"):
                learner.decays = self.float_2
            with self.assertRaisesRegex(AttributeError, "gofit is readonly in HawkesExpKern"):
                learner.gofit = gofit

        decay_array = np.random.rand(len(self.events), len(self.events))
        learner = HawkesExpKern(decay_array, gofit="least-squares")
        np.testing.assert_array_equal(learner._model_obj.decays, decay_array)
        with self.assertRaisesRegex(NotImplementedError, "constant decay"):
            HawkesExpKern(decay_array, gofit="likelihood")

        learner = HawkesExpKern(self.decays, max_iter=-1).fit(self.events, start=0.2)
        simu = learner._corresponding_simu()
        self.assertIsInstance(simu, SimuHawkesExpKernels)
        self.assertEqual(simu.decays, learner.decays)
        np.testing.assert_array_equal(simu.baseline, learner.baseline)
        np.testing.assert_array_equal(simu.adjacency, learner.adjacency)


class HawkesSumExpKernParityTest(ParametricLearnerParityMixin, unittest.TestCase):
    penalties = SUMEXP_PENALTIES

    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230
        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]
        self.n_decays = 2
        self.decays = np.random.rand(self.n_decays)

    def make_learner(self, **kwargs):
        return HawkesSumExpKern(self.decays, **kwargs)

    def test_fit_start_matches_tick_start_contract(self):
        n_nodes = len(self.events)
        n_coeffs = n_nodes + n_nodes * n_nodes * self.n_decays
        learner = HawkesSumExpKern(self.decays, max_iter=-1)
        learner.fit(self.events)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs))
        learner.fit(self.events, start=self.float_1)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs) * self.float_1)
        learner.fit(self.events, start=self.int_1)
        np.testing.assert_array_equal(learner.coeffs, np.ones(n_coeffs) * self.int_1)
        random_coeffs = np.random.rand(n_coeffs)
        learner.fit(self.events, start=random_coeffs)
        np.testing.assert_array_equal(learner.coeffs, random_coeffs)

    def test_score_matches_tick_seeded_reference(self):
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
        learner = HawkesSumExpKern(self.decays, record_every=1)
        with self.assertRaisesRegex(ValueError, r"^You must either call `fit` before `score` or provide events$"):
            learner.score()
        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes, self.n_decays)
        learner.fit(train_events)
        self.assertAlmostEqual(learner.score(), 1.684827141, delta=1e-4)
        self.assertAlmostEqual(
            learner.score(baseline=given_baseline, adjacency=given_adjacency),
            1.16247892,
            delta=1e-8,
        )
        self.assertAlmostEqual(learner.score(test_events), 1.66494295, delta=1e-4)
        self.assertAlmostEqual(
            learner.score(test_events, baseline=given_baseline, adjacency=given_adjacency),
            1.1081362,
            delta=1e-8,
        )

    def test_fit_and_sparse_events_run_representative_paths(self):
        for solver, penalty in [("gd", "none"), ("agd", "l2"), ("svrg", "l1"), ("bfgs", "l2"), ("sgd", "elasticnet")]:
            kwargs = dict(solver=solver, penalty=penalty, max_iter=2, verbose=False)
            if penalty != "none":
                kwargs["C"] = 50
            if solver == "sgd":
                kwargs["random_state"] = 179312
                kwargs["step"] = 1e-5
            elif solver == "svrg":
                kwargs["random_state"] = 179312
                kwargs["step"] = 1e-5
            learner = HawkesSumExpKern(self.decays, **kwargs).fit(self.events, start=0.01)
            self.assertEqual(learner.adjacency.shape, (3, 3, self.n_decays))
            self.assertTrue(np.all(np.isfinite(learner.coeffs)))

        sparse_events = [
            [np.array([0.2, 0.7, 1.4]), np.array([], dtype=float)],
            [np.array([0.1, 0.4]), np.array([0.9, 1.8])],
        ]
        learner = HawkesSumExpKern(self.decays, max_iter=5, verbose=False).fit(sparse_events)
        self.assertEqual(learner.baseline.shape, (2,))
        self.assertEqual(learner.adjacency.shape, (2, 2, self.n_decays))
        self.assertTrue(np.isfinite(learner.score()))

    def test_settings_model_penalty_solver_and_corresponding_simu(self):
        self.assert_common_solver_settings(self.make_learner)
        self.assert_common_penalty_settings(self.make_learner)
        self.assert_common_elastic_net_settings(self.make_learner)
        self.assert_common_solver_step_and_random_state(self.make_learner)

        with self.assertRaisesRegex(ValueError, r"^``solver`` must be one of agd, bfgs, gd, sgd, svrg, got wrong_name$"):
            HawkesSumExpKern(self.decays, solver="wrong_name")
        with self.assertRaisesRegex(ValueError, r"^``penalty`` must be one of elasticnet, l1, l2, none, got wrong_name$"):
            HawkesSumExpKern(self.decays, penalty="wrong_name")

        learner = HawkesSumExpKern(self.decays)
        np.testing.assert_array_equal(learner.decays, self.decays)
        self.assertIsInstance(learner._model_obj, ModelHawkesSumExpKernLeastSq)
        np.testing.assert_array_equal(learner._model_obj.decays, self.decays)
        self.assertEqual(learner.n_baselines, 1)
        self.assertIsNone(learner.period_length)

        learner = HawkesSumExpKern(self.decays, n_baselines=3, period_length=2.0)
        self.assertEqual(learner._model_obj.n_baselines, 3)
        self.assertEqual(learner._model_obj.period_length, 2.0)
        with self.assertRaisesRegex(AttributeError, "decays is readonly in HawkesSumExpKern"):
            learner.decays = self.decays + 1
        with self.assertRaisesRegex(AttributeError, "n_baselines is readonly in HawkesSumExpKern"):
            learner.n_baselines = 4
        with self.assertRaisesRegex(AttributeError, "period_length is readonly in HawkesSumExpKern"):
            learner.period_length = 3.0
        with self.assertRaisesRegex(ValueError, "You must fit data before getting estimated baseline"):
            learner.baseline
        with self.assertRaisesRegex(ValueError, "You must fit data before getting estimated adjacency"):
            learner.adjacency

        learner = HawkesSumExpKern(self.decays, max_iter=-1).fit(self.events, start=0.2)
        simu = learner._corresponding_simu()
        self.assertIsInstance(simu, SimuHawkesSumExpKernels)
        np.testing.assert_array_equal(simu.decays, learner.decays)
        np.testing.assert_array_equal(simu.baseline, learner.baseline)
        np.testing.assert_array_equal(simu.adjacency, learner.adjacency)

        learner = HawkesSumExpKern(self.decays, n_baselines=3, period_length=1.0, max_iter=-1)
        learner.fit(self.events, start=0.2)
        simu = learner._corresponding_simu()
        np.testing.assert_array_equal(simu.baseline, learner.baseline)
        self.assertEqual(simu.period_length, learner.period_length)


if __name__ == "__main__":
    unittest.main()
