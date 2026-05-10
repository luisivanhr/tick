"""Simulate homogeneous and inhomogeneous Poisson processes."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.base import TimeFunction
from our_hawkes.hawkes import SimuInhomogeneousPoisson, SimuPoissonProcess
from our_hawkes.plot import plot_point_process, plot_timefunction


def main():
    poisson = SimuPoissonProcess([0.4, 0.8], end_time=20.0, seed=7, verbose=False)
    poisson.track_intensity(0.1)
    poisson.simulate()

    t_values = np.linspace(0.0, 20.0, 25)
    intensity = 0.25 + 0.5 * (1.0 + np.sin(t_values / 2.0))
    time_function = TimeFunction((t_values, intensity))
    inhomogeneous = SimuInhomogeneousPoisson([time_function], end_time=20.0, seed=8, verbose=False)
    inhomogeneous.track_intensity(0.1)
    inhomogeneous.simulate()

    fig_poisson = plot_point_process(poisson, plot_intensity=True, show=False)
    fig_inhomogeneous = plot_point_process(inhomogeneous, plot_intensity=True, show=False)
    fig_time_function = plot_timefunction(time_function, show=False)

    print("homogeneous jumps:", poisson.n_total_jumps)
    print("inhomogeneous jumps:", inhomogeneous.n_total_jumps)

    plt.close(fig_poisson)
    plt.close(fig_inhomogeneous)
    plt.close(fig_time_function)


if __name__ == "__main__":
    main()
