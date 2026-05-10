"""Fit sparse Hawkes adjacency with ADM4."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesADM4, SimuHawkesExpKernels, SimuHawkesMulti
from our_hawkes.plot import plot_hawkes_kernel_norms


def main():
    decay = 1.5
    baseline = np.array([0.18, 0.12, 0.16])
    adjacency = np.array(
        [
            [0.12, 0.00, 0.04],
            [0.03, 0.10, 0.00],
            [0.00, 0.04, 0.12],
        ]
    )

    base = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=decay,
        baseline=baseline,
        end_time=80.0,
        seed=1039,
        verbose=False,
    )
    multi = SimuHawkesMulti(base, n_simulations=3, n_threads=1).simulate()

    learner = HawkesADM4(decay=decay, C=50.0, max_iter=20, verbose=False)
    learner.fit(multi.timestamps, end_times=multi.end_time)

    fig = plot_hawkes_kernel_norms(learner, show=False)

    print("baseline:", np.round(learner.baseline, 6))
    print("adjacency:", np.round(learner.adjacency, 6))

    plt.close(fig)


if __name__ == "__main__":
    main()
