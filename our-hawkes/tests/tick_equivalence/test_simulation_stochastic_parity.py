"""Source-backed parity checks for tick's remaining simulation tests.

The assertions in this file are ported from:

- tick/hawkes/simulation/tests/simu_hawkes_test.py
- tick/hawkes/simulation/tests/simu_inhomogeneous_poisson_test.py
- tick/hawkes/simulation/tests/simu_point_process_test.py
- tick/hawkes/simulation/tests/simu_poisson_test.py

These tests preserve tick's stochastic assertions instead of claiming exact
C++ RNG stream parity.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from our_hawkes.base import TimeFunction  # noqa: E402
from our_hawkes.hawkes import (  # noqa: E402
    HawkesKernelExp,
    SimuHawkes,
    SimuInhomogeneousPoisson,
    SimuPoissonProcess,
)


class SimulationStochasticParityTest(unittest.TestCase):
    def setUp(self):
        confidence = 1e-4
        self.z = norm.ppf(1 - confidence / 2)

    def test_simu_hawkes_force_simulation(self):
        hawkes = SimuHawkes(kernels=[[HawkesKernelExp(2, 3)]], baseline=[1], verbose=False)
        hawkes.end_time = 10

        with self.assertRaisesRegex(
            ValueError,
            r"^Simulation not launched as this Hawkes process is not stable "
            r"\(spectral radius of 2\). You can use force_simulation parameter "
            r"if you really want to simulate it$",
        ):
            hawkes.simulate()

        with self.assertWarnsRegex(
            UserWarning,
            r"^This process has already been simulated until time 0.000000$",
        ):
            hawkes.end_time = 0
            hawkes.force_simulation = True
            hawkes.simulate()

    def test_simulation_1d_inhomogeneous_poisson(self):
        run_time = 30
        t_values = np.linspace(0, run_time - 3, 100)
        y_values = np.maximum(0.5 + np.sin(t_values), 0)

        tf = TimeFunction((t_values, y_values))
        tf_zero = TimeFunction(0)
        process = SimuInhomogeneousPoisson([tf, tf_zero], seed=2937, end_time=run_time, verbose=False)
        process.simulate()

        timestamps = process.timestamps
        self.assertGreater(len(timestamps[0]), 2)
        self.assertEqual(len(timestamps[1]), 0)
        self.assertEqual(np.prod(tf.value(timestamps[0]) > 0), 1)

    def test_simulation_time(self):
        process = SimuPoissonProcess([1, 3, 2.2], end_time=30, verbose=False)
        self.assertEqual(process.simulation_time, 0)
        process.simulate()
        self.assertEqual(process.simulation_time, 30)

    def test_simulation_1d_poisson(self):
        lambda_0 = 2.9
        time = 1000.0

        process = SimuPoissonProcess(lambda_0, seed=139, end_time=time, verbose=False)
        process.simulate()
        n_total_jumps = process.n_total_jumps
        tcl = (n_total_jumps - time * lambda_0) / np.sqrt(time * lambda_0)
        self.assertLess(np.abs(tcl), self.z)

    def test_simulation_nd_poisson(self):
        lambdas = np.array([1.0, 2.0, 3.3])
        time = 1000.0

        process = SimuPoissonProcess(lambdas, seed=13923, end_time=time, verbose=False)
        process.simulate()

        jumps = np.array(list(map(len, process.timestamps)))
        tcl = np.divide(jumps - time * lambdas, np.sqrt(time * lambdas))
        self.assertEqual(sum(np.abs(tcl) > self.z), 0)


if __name__ == "__main__":
    unittest.main()
