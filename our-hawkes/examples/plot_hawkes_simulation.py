"""Simulate a small Hawkes process with exponential kernels."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import SimuHawkesExpKernels
from our_hawkes.plot import plot_hawkes_kernel_norms, plot_point_process


def main():
    adjacency = np.array([[0.25, 0.05], [0.10, 0.20]])
    decays = np.array([[1.5, 2.0], [1.0, 1.8]])
    baseline = np.array([0.4, 0.3])

    hawkes = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=decays,
        baseline=baseline,
        end_time=30.0,
        seed=123,
        verbose=False,
    )
    hawkes.track_intensity(0.1)
    hawkes.simulate()

    fig_process = plot_point_process(hawkes, plot_intensity=True, show=False)
    fig_norms = plot_hawkes_kernel_norms(hawkes, show=False)

    print("jumps:", hawkes.n_total_jumps)
    print("spectral radius:", round(hawkes.spectral_radius(), 6))
    print("mean intensity:", np.round(hawkes.mean_intensity(), 6))

    plt.close(fig_process)
    plt.close(fig_norms)


if __name__ == "__main__":
    main()
