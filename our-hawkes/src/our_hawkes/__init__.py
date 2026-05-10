"""Pure-Python Hawkes and point-process tools."""

from .base import BaseEstimator, History, TimeFunction
from .plot import (
    plot_basis_kernels,
    plot_estimated_intensity,
    plot_hawkes_baseline_and_kernels,
    plot_hawkes_kernel_norms,
    plot_hawkes_kernels,
    plot_point_process,
    plot_timefunction,
    qq_plots,
)

__all__ = [
    "BaseEstimator",
    "History",
    "TimeFunction",
    "plot_point_process",
    "plot_hawkes_kernels",
    "plot_hawkes_kernel_norms",
    "plot_hawkes_baseline_and_kernels",
    "plot_basis_kernels",
    "plot_estimated_intensity",
    "plot_timefunction",
    "qq_plots",
]

__version__ = "0.1.0"
