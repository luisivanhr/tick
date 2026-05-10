"""Fit non-parametric Hawkes kernels with EM."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.base import TimeFunction
from our_hawkes.hawkes import HawkesEM, HawkesKernelExp, HawkesKernelTimeFunc, SimuHawkes
from our_hawkes.plot import plot_hawkes_kernels


def main():
    t_values = np.array([0.0, 0.8, 1.5, 2.4, 3.0])
    y_values = np.array([0.0, 0.18, 0.03, 0.08, 0.0])
    kernel_time_func = HawkesKernelTimeFunc(
        TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight, dt=0.1)
    )

    hawkes = SimuHawkes(
        baseline=np.array([0.2, 0.25]),
        end_time=80.0,
        max_jumps=500,
        seed=2334,
        verbose=False,
    )
    hawkes.set_kernel(0, 0, kernel_time_func)
    hawkes.set_kernel(0, 1, HawkesKernelExp(0.25, 0.7))
    hawkes.simulate()

    learner = HawkesEM(kernel_support=3.0, kernel_size=12, max_iter=8, tol=1e-4, verbose=False)
    learner.fit(hawkes.timestamps, end_times=hawkes.end_time)

    fig = plot_hawkes_kernels(learner, hawkes=hawkes, support=3.0, show=False)

    print("baseline:", np.round(learner.baseline, 6))
    print("kernel norms:", np.round(learner.get_kernel_norms(), 6))

    plt.close(fig)


if __name__ == "__main__":
    main()
