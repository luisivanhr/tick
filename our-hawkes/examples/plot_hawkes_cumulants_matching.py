"""Estimate Hawkes kernel norms with cumulant matching."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesCumulantMatching, SimuHawkesExpKernels, SimuHawkesMulti
from our_hawkes.plot import plot_hawkes_kernel_norms


def main():
    baseline = np.array([0.25, 0.22, 0.18])
    adjacency = np.array(
        [
            [0.12, 0.04, 0.00],
            [0.02, 0.10, 0.03],
            [0.00, 0.05, 0.08],
        ]
    )
    decays = np.array(
        [
            [1.0, 1.5, 1.0],
            [1.2, 1.1, 1.4],
            [1.0, 1.3, 1.2],
        ]
    )
    base = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=80.0,
        seed=7168,
        verbose=False,
    )
    multi = SimuHawkesMulti(base, n_simulations=4, n_threads=1).simulate()

    learner = HawkesCumulantMatching(integration_support=5.0, max_iter=50, verbose=False)
    learner.fit(multi.timestamps, end_times=multi.end_time)

    fig = plot_hawkes_kernel_norms(learner, show=False)

    print("mean intensity:", np.round(learner.mean_intensity, 6))
    print("estimated norms:", np.round(learner.get_kernel_norms(), 6))
    print("objective:", round(learner.objective(), 6))

    plt.close(fig)


if __name__ == "__main__":
    main()
