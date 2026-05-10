"""Fit a fixed-decay exponential Hawkes model."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesExpKern, SimuHawkesExpKernels


def main():
    simu = SimuHawkesExpKernels(
        adjacency=np.array([[0.20, 0.05], [0.10, 0.15]]),
        decays=1.2,
        baseline=np.array([0.5, 0.4]),
        end_time=30.0,
        seed=3,
        verbose=False,
    )
    simu.simulate()

    learner = HawkesExpKern(
        decays=1.2,
        gofit="likelihood",
        penalty="l2",
        C=100.0,
        max_iter=25,
        verbose=False,
    ).fit(simu.timestamps, end_times=simu.end_time)

    print("baseline:", np.round(learner.baseline, 6))
    print("adjacency:", np.round(learner.adjacency, 6))
    print("score:", round(learner.score(), 6))


if __name__ == "__main__":
    main()
