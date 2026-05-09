"""Pure-Python Hawkes and point-process tools."""

from .base import BaseEstimator, History, TimeFunction
from .plot import (
    plot_hawkes_kernel_norms,
    plot_hawkes_kernels,
    plot_point_process,
    qq_plots,
)

__all__ = [
    "BaseEstimator",
    "History",
    "TimeFunction",
    "plot_point_process",
    "plot_hawkes_kernels",
    "plot_hawkes_kernel_norms",
    "qq_plots",
]

__version__ = "0.1.0"
