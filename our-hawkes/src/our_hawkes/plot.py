"""Plotting helpers for point processes and Hawkes learners."""

from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_point_process(
    point_process,
    plot_intensity: bool = False,
    n_points: int = 10000,
    plot_nodes=None,
    t_min: float | None = None,
    t_max: float | None = None,
    max_jumps: int | None = None,
    show: bool = True,
    ax=None,
):
    """Plot timestamps and optionally tracked intensities."""

    plt = _plt()
    timestamps = point_process.timestamps
    nodes = list(range(len(timestamps))) if plot_nodes is None else list(plot_nodes)
    if t_min is None:
        t_min = 0.0
    if t_max is None:
        t_max = point_process.simulation_time
        if isinstance(t_max, list):
            t_max = max(t_max)

    if ax is None:
        n_rows = 2 if plot_intensity else 1
        _, axes = plt.subplots(n_rows, 1, sharex=True, squeeze=False)
        axes = axes[:, 0]
    elif isinstance(ax, (list, tuple, np.ndarray)):
        axes = ax
    else:
        axes = [ax]

    stamp_ax = axes[0]
    for y, node in enumerate(nodes):
        ts = np.asarray(timestamps[node], dtype=float)
        ts = ts[(ts >= t_min) & (ts <= t_max)]
        if max_jumps is not None:
            ts = ts[:max_jumps]
        stamp_ax.vlines(ts, y + 0.1, y + 0.9, linewidth=0.8)
    stamp_ax.set_yticks(np.arange(len(nodes)) + 0.5, [str(n) for n in nodes])
    stamp_ax.set_ylabel("node")
    stamp_ax.set_xlim(t_min, t_max)

    if plot_intensity:
        if not point_process.is_intensity_tracked():
            step = (t_max - t_min) / max(n_points, 1)
            point_process.track_intensity(step)
            point_process.set_timestamps(timestamps, end_time=t_max)
        intensity_ax = axes[1]
        times = point_process.intensity_tracked_times
        for node in nodes:
            intensity_ax.plot(times, point_process.tracked_intensity[node], label=str(node))
        intensity_ax.set_ylabel("intensity")
        intensity_ax.legend(loc="best")
    axes[-1].set_xlabel("time")
    if show:
        plt.show()
    return axes[0].figure


def plot_hawkes_kernels(learner, support=None, n_points: int = 200, show: bool = True, ax=None):
    """Plot all estimated Hawkes kernels."""

    plt = _plt()
    supports = learner.get_kernel_supports()
    n_nodes = supports.shape[0]
    if ax is None:
        _, axes = plt.subplots(n_nodes, n_nodes, squeeze=False)
    else:
        axes = np.asarray(ax).reshape((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            local_support = float(support if support is not None else supports[i, j])
            if local_support <= 0:
                local_support = 1.0
            x = np.linspace(0.0, local_support, n_points)
            axes[i, j].plot(x, learner.get_kernel_values(i, j, x))
            axes[i, j].set_title(f"{i},{j}")
    if show:
        plt.show()
    return axes[0, 0].figure


def plot_hawkes_kernel_norms(learner, show: bool = True, ax=None):
    plt = _plt()
    norms = learner.get_kernel_norms()
    if ax is None:
        _, ax = plt.subplots()
    image = ax.imshow(norms, origin="upper")
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    ax.figure.colorbar(image, ax=ax)
    if show:
        plt.show()
    return ax.figure


def qq_plots(point_process, show: bool = True, ax=None):
    """Plot exponential QQ diagnostics from stored compensator values."""

    plt = _plt()
    if not point_process.tracked_compensator:
        point_process.store_compensator_values()
    if ax is None:
        _, ax = plt.subplots()
    residuals = []
    for values in point_process.tracked_compensator:
        if len(values) > 1:
            residuals.extend(np.diff(values))
    residuals = np.sort(np.asarray(residuals, dtype=float))
    if residuals.size:
        probs = (np.arange(residuals.size) + 0.5) / residuals.size
        theoretical = -np.log1p(-probs)
        ax.scatter(theoretical, residuals, s=12)
        lim = max(float(np.max(theoretical)), float(np.max(residuals)))
        ax.plot([0, lim], [0, lim], color="black", linewidth=1)
    ax.set_xlabel("theoretical exponential quantile")
    ax.set_ylabel("empirical residual quantile")
    if show:
        plt.show()
    return ax.figure
