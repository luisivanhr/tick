"""Estimate Hawkes kernels with the conditional-law learner."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesConditionalLaw, HawkesKernelPowerLaw, SimuHawkes
from our_hawkes.plot import plot_hawkes_kernels


def main():
    support = 25.0
    kernels = [
        [
            HawkesKernelPowerLaw(0.08, 0.5, 1.4, support=support),
            HawkesKernelPowerLaw(0.05, 0.5, 1.4, support=support),
        ],
        [
            HawkesKernelPowerLaw(0.03, 0.5, 1.4, support=support),
            HawkesKernelPowerLaw(0.06, 0.5, 1.4, support=support),
        ],
    ]
    hawkes = SimuHawkes(
        kernels=kernels,
        baseline=np.array([0.12, 0.10]),
        end_time=160.0,
        max_jumps=600,
        seed=382,
        verbose=False,
    )
    hawkes.simulate()

    learner = HawkesConditionalLaw(
        claw_method="log",
        delta_lag=0.2,
        min_lag=0.05,
        max_lag=20.0,
        quad_method="log",
        n_quad=25,
        min_support=0.05,
        max_support=support,
        n_threads=1,
    )
    learner.fit(hawkes.timestamps, end_times=hawkes.end_time)

    fig = plot_hawkes_kernels(
        learner,
        hawkes=hawkes,
        log_scale=True,
        min_support=0.05,
        support=20.0,
        show=False,
    )

    print("mean intensity:", np.round(learner.mean_intensity, 6))
    print("kernel norms:", np.round(learner.get_kernel_norms(), 6))

    plt.close(fig)


if __name__ == "__main__":
    main()
