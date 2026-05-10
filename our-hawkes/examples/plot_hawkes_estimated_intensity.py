"""Fit a Hawkes model and plot estimated versus simulated intensity."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesSumExpKern, SimuHawkesSumExpKernels
from our_hawkes.plot import plot_estimated_intensity, plot_point_process, qq_plots


def main():
    decays = np.array([0.6, 1.5])
    adjacency = np.array(
        [
            [[0.05, 0.10], [0.03, 0.04]],
            [[0.02, 0.03], [0.10, 0.05]],
        ]
    )
    baseline = np.array([0.18, 0.12])

    model = SimuHawkesSumExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=80.0,
        seed=1039,
        verbose=False,
    )
    model.track_intensity(0.1)
    model.simulate()

    learner = HawkesSumExpKern(
        decays=decays,
        penalty="elasticnet",
        C=50.0,
        elastic_net_ratio=0.8,
        max_iter=30,
        verbose=False,
    ).fit(model.timestamps, end_times=model.end_time)

    fig_estimated = plot_estimated_intensity(
        learner,
        model.timestamps,
        intensity_track_step=0.1,
        end_time=model.end_time,
        t_min=10.0,
        t_max=40.0,
        show=False,
    )
    fig_simulated = plot_point_process(model, plot_intensity=True, t_min=10.0, t_max=40.0, show=False)

    model.store_compensator_values()
    fig_qq = qq_plots(model, show=False)

    print("fitted baseline:", np.round(learner.baseline, 6))
    print("fitted kernel norms:", np.round(learner.get_kernel_norms(), 6))

    plt.close(fig_estimated)
    plt.close(fig_simulated)
    plt.close(fig_qq)


if __name__ == "__main__":
    main()
