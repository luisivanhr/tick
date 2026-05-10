"""Plotting helpers for point processes and Hawkes learners."""

from __future__ import annotations

import numpy as np

from our_hawkes.base import TimeFunction


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _as_axes(ax, shape, plt, figsize=None, sharex=False, sharey=False):
    if ax is None:
        _, axes = plt.subplots(*shape, squeeze=False, figsize=figsize, sharex=sharex, sharey=sharey)
        return axes, True
    axes = np.asarray(ax, dtype=object)
    if axes.shape == ():
        axes = axes.reshape((1, 1))
    elif axes.ndim == 1 and shape[1] == 1:
        axes = axes.reshape((shape[0], 1))
    elif axes.ndim == 1 and shape[0] == 1:
        axes = axes.reshape((1, shape[1]))
    if axes.shape != shape:
        raise ValueError(f"ax has shape {axes.shape}, expected {shape}")
    return axes, False


def _node_names(n_nodes, plot_nodes, node_names):
    if node_names is None:
        return [f"ticks #{node}" for node in plot_nodes]
    if len(node_names) != len(plot_nodes):
        raise ValueError(f"node_names must have length {len(plot_nodes)}, got {len(node_names)}")
    return list(node_names)


def _simulation_end_time(point_process):
    end_time = getattr(point_process, "end_time", None)
    if isinstance(end_time, (list, tuple, np.ndarray)):
        return float(np.max(end_time))
    if end_time is not None:
        return float(end_time)
    return float(getattr(point_process, "simulation_time", 0.0))


def _hawkes_n_nodes(obj):
    n_nodes = getattr(obj, "n_nodes", None)
    if isinstance(n_nodes, (list, tuple, np.ndarray)):
        return int(n_nodes[0])
    if n_nodes is not None:
        return int(n_nodes)
    return int(np.asarray(obj.kernels, dtype=object).shape[0])


def _kernel_supports(obj):
    if hasattr(obj, "get_kernel_supports"):
        return np.asarray(obj.get_kernel_supports(), dtype=float)
    kernels = np.asarray(obj.kernels, dtype=object)
    return np.vectorize(lambda kernel: kernel.get_plot_support(), otypes=[float])(kernels)


def _kernel_norms(obj):
    if hasattr(obj, "get_kernel_norms"):
        return np.asarray(obj.get_kernel_norms(), dtype=float)
    kernels = np.asarray(obj.kernels, dtype=object)
    return np.vectorize(lambda kernel: kernel.get_norm(), otypes=[float])(kernels)


def _kernel_values(obj, i, j, x):
    if hasattr(obj, "get_kernel_values"):
        return obj.get_kernel_values(i, j, x)
    return obj.kernels[i, j].get_values(x)


def _interval_mask(values, t_min, t_max):
    values = np.asarray(values, dtype=float)
    return (values >= t_min) & (values <= t_max)


def plot_point_process(
    point_process,
    plot_intensity=None,
    n_points: int = 10000,
    plot_nodes=None,
    node_names=None,
    t_min: float | None = None,
    t_max: float | None = None,
    max_jumps: int | None = None,
    show: bool = True,
    ax=None,
):
    """Plot point-process timestamps and, optionally, tracked intensities."""

    plt = _plt()
    n_nodes = int(getattr(point_process, "n_nodes", len(point_process.timestamps)))
    nodes = list(range(n_nodes)) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(n_nodes, nodes, node_names)

    end_time = _simulation_end_time(point_process)
    t_min = 0.0 if t_min is None else float(t_min)
    t_max = end_time if t_max is None else float(t_max)
    if t_min >= end_time and end_time > 0:
        raise ValueError("t_min must be smaller than the process end time")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min")

    if plot_intensity is None:
        plot_intensity = bool(point_process.is_intensity_tracked())

    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(10, max(2.5, 2.4 * len(nodes))),
        sharex=True,
        sharey=False,
    )
    if not created:
        show = False

    timestamps = point_process.timestamps
    if plot_intensity and not point_process.is_intensity_tracked():
        step = (t_max - t_min) / max(int(n_points), 1)
        point_process.track_intensity(step)
        point_process.set_timestamps(timestamps, end_time=t_max)

    intensity_times = None
    if plot_intensity:
        intensity_times = np.asarray(point_process.intensity_tracked_times, dtype=float)
        time_mask = _interval_mask(intensity_times, t_min, t_max)
        intensity_times = intensity_times[time_mask]

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        ts = np.asarray(timestamps[node], dtype=float)
        ts = ts[_interval_mask(ts, t_min, t_max)]
        if max_jumps is not None:
            ts = ts[: int(max_jumps)]

        if plot_intensity:
            intensity = np.asarray(point_process.tracked_intensity[node], dtype=float)
            intensity = intensity[time_mask]
            if intensity_times.size:
                x = np.linspace(float(intensity_times[0]), float(intensity_times[-1]), int(n_points))
                y = np.interp(x, intensity_times, intensity)
                axis.plot(x, y, label="intensity")
                if ts.size:
                    axis.scatter(ts, np.interp(ts, intensity_times, intensity), s=18, label="jumps")
            axis.set_ylabel(label)
        else:
            axis.vlines(ts, 0.0, 1.0, linewidth=0.9)
            axis.set_yticks([])
            axis.set_ylabel(label)
        axis.set_xlim(t_min, t_max)
        if plot_intensity:
            axis.legend(loc="best")

    axes[-1, 0].set_xlabel("time")
    if show:
        plt.show()
    return axes[0, 0].figure


def plot_hawkes_kernels(
    kernel_object,
    support=None,
    hawkes=None,
    n_points: int = 300,
    show: bool = True,
    log_scale: bool = False,
    min_support: float = 1e-4,
    ax=None,
):
    """Plot every entry of a Hawkes kernel matrix."""

    plt = _plt()
    supports = _kernel_supports(kernel_object)
    n_nodes = _hawkes_n_nodes(kernel_object)
    if support is None or support <= 0:
        finite_supports = supports[np.isfinite(supports) & (supports > 0)]
        support = float(np.max(finite_supports)) if finite_supports.size else 1.0
        support *= 1.2

    axes, created = _as_axes(
        ax,
        (n_nodes, n_nodes),
        plt,
        figsize=(3.2 * n_nodes, 2.7 * n_nodes),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False

    if log_scale:
        x = np.logspace(np.log10(min_support), np.log10(support), int(n_points))
    else:
        x = np.linspace(0.0, float(support), int(n_points))

    for i in range(n_nodes):
        for j in range(n_nodes):
            axis = axes[i, j]
            axis.plot(x, _kernel_values(kernel_object, i, j, x), label=f"kernel ({i}, {j})")
            if hawkes is not None:
                axis.plot(x, hawkes.kernels[i, j].get_values(x), linestyle="--", label=f"true ({i}, {j})")
            if i == n_nodes - 1:
                axis.set_xlabel("time")
            axis.set_ylabel(f"phi[{i},{j}]")
            if log_scale:
                axis.set_xscale("log")
                axis.set_yscale("log")
            axis.legend(loc="best", fontsize=8)

    if show:
        plt.show()
    return axes[0, 0].figure


def plot_hawkes_kernel_norms(
    kernel_object,
    show: bool = True,
    pcolor_kwargs=None,
    node_names=None,
    rotate_x_labels: float = 0.0,
    ax=None,
):
    """Plot the matrix of Hawkes kernel norms."""

    plt = _plt()
    norms = _kernel_norms(kernel_object)
    n_nodes = norms.shape[0]
    labels = list(range(n_nodes)) if node_names is None else list(node_names)
    if len(labels) != n_nodes:
        raise ValueError(f"node_names must have length {n_nodes}, got {len(labels)}")
    if pcolor_kwargs is None:
        pcolor_kwargs = {}
    if norms.size and np.nanmin(norms) < 0:
        vmax = float(np.nanmax(np.abs(norms)))
        pcolor_kwargs.setdefault("cmap", "RdBu_r")
        pcolor_kwargs.setdefault("vmin", -vmax)
        pcolor_kwargs.setdefault("vmax", vmax)
    else:
        pcolor_kwargs.setdefault("cmap", "Blues")

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.2))
    else:
        show = False
    image = ax.imshow(norms, origin="upper", **pcolor_kwargs)
    ax.set_xticks(np.arange(n_nodes), [f"{label}" for label in labels], rotation=-rotate_x_labels)
    ax.set_yticks(np.arange(n_nodes), [f"{label}" for label in labels])
    ax.xaxis.tick_top()
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    ax.figure.colorbar(image, ax=ax)
    if show:
        plt.show()
    return ax.figure


def _baseline_values(hawkes_object, node, t_values):
    if hasattr(hawkes_object, "get_baseline_values"):
        return hawkes_object.get_baseline_values(node, t_values)
    baseline = np.asarray(hawkes_object.baseline, dtype=float)
    if baseline.ndim == 1:
        return np.full_like(t_values, baseline[node], dtype=float)
    period_length = getattr(hawkes_object, "period_length", None)
    if period_length is None:
        raise ValueError("period_length is required for piecewise baseline arrays")
    idx = np.floor(((t_values % period_length) / period_length) * baseline.shape[1]).astype(int)
    idx = np.minimum(idx, baseline.shape[1] - 1)
    return baseline[node, idx]


def plot_hawkes_baseline_and_kernels(
    hawkes_object,
    kernel_support=None,
    hawkes=None,
    n_points: int = 300,
    show: bool = True,
    log_scale: bool = False,
    min_support: float = 1e-4,
    ax=None,
):
    """Plot Hawkes baselines in the first column and kernels beside them."""

    plt = _plt()
    n_nodes = int(hawkes_object.n_nodes)
    axes, created = _as_axes(
        ax,
        (n_nodes, n_nodes + 1),
        plt,
        figsize=(3.0 * (n_nodes + 1), 2.7 * n_nodes),
        sharex=False,
        sharey=False,
    )
    if not created:
        show = False

    kernel_axes = axes[:, 1:]
    plot_hawkes_kernels(
        hawkes_object,
        support=kernel_support,
        hawkes=hawkes,
        n_points=n_points,
        show=False,
        log_scale=log_scale,
        min_support=min_support,
        ax=kernel_axes,
    )

    period = getattr(hawkes_object, "period_length", None)
    if period is None:
        period = getattr(hawkes, "period_length", None) if hawkes is not None else None
    if period is None:
        period = kernel_support
    if period is None or period <= 0:
        supports = _kernel_supports(hawkes_object)
        finite_supports = supports[np.isfinite(supports) & (supports > 0)]
        period = float(np.max(finite_supports)) if finite_supports.size else 1.0

    t_values = np.linspace(0.0, float(period), int(n_points))
    for i in range(n_nodes):
        axis = axes[i, 0]
        axis.plot(t_values, _baseline_values(hawkes_object, i, t_values), label=f"baseline ({i})")
        if hawkes is not None:
            axis.plot(
                t_values,
                _baseline_values(hawkes, i, t_values),
                linestyle="--",
                label=f"true baseline ({i})",
            )
        axis.set_xlabel("time")
        axis.set_ylabel(f"mu[{i}]")
        axis.legend(loc="best", fontsize=8)

    if show:
        plt.show()
    return axes[0, 0].figure


def _normalize_functions(y_values_list, t_values):
    y_values = np.asarray(y_values_list, dtype=float)
    normalizations = []
    for values in y_values:
        integral = float(np.trapezoid(values, t_values))
        normalizations.append(1.0 / integral if abs(integral) > 1e-15 else 1.0)
    return (y_values.T * normalizations).T, np.asarray(normalizations, dtype=float)


def _find_best_match(diff_matrix):
    diff_matrix = np.asarray(diff_matrix, dtype=float).copy()
    matches = []
    for _ in range(diff_matrix.shape[0]):
        row, col = np.unravel_index(np.argmin(diff_matrix), diff_matrix.shape)
        matches.append((row, col))
        diff_matrix[row, :] = np.inf
        diff_matrix[:, col] = np.inf
    return matches


def _piecewise_step_xy(discretization, values):
    values = np.asarray(values, dtype=float)
    edges = np.asarray(discretization, dtype=float)
    return np.hstack((edges[0], np.repeat(edges[1:-1], 2), edges[-1])), np.repeat(values, 2)


def plot_basis_kernels(learner, support=None, basis_kernels=None, n_points: int = 300, show: bool = True, ax=None):
    """Plot basis kernels from a :class:`HawkesBasisKernels` learner."""

    plt = _plt()
    if support is None or support <= 0:
        support = learner.kernel_support
    axes, created = _as_axes(
        ax,
        (1, learner.n_basis),
        plt,
        figsize=(3.2 * learner.n_basis, 3.0),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False
    axes = axes[0]

    matches = [(i, i) for i in range(learner.n_basis)]
    true_values = None
    true_normalizations = None
    estimated_normalizations = np.ones(learner.n_basis)
    if basis_kernels is not None:
        if len(basis_kernels) != learner.n_basis:
            raise ValueError(f"learner has {learner.n_basis} basis kernels, got {len(basis_kernels)}")
        t_grid = learner.kernel_discretization[:-1]
        true_values = np.asarray([fn(t_grid) for fn in basis_kernels], dtype=float)
        normalized_true, true_normalizations = _normalize_functions(true_values, t_grid)
        normalized_estimated, estimated_normalizations = _normalize_functions(learner.basis_kernels, t_grid)
        diff = np.array(
            [
                [np.trapezoid(np.abs(est - true), t_grid) for true in normalized_true]
                for est in normalized_estimated
            ]
        )
        matches = _find_best_match(diff)

    dense_t = np.linspace(0.0, float(support), int(n_points))
    for estimated_index, basis_index in matches:
        axis = axes[basis_index]
        step_t, step_y = _piecewise_step_xy(learner.kernel_discretization, learner.basis_kernels[estimated_index])
        axis.step(step_t, step_y, where="post", label=f"estimated {estimated_index}")
        if basis_kernels is not None:
            scale = true_normalizations[basis_index] / estimated_normalizations[estimated_index]
            axis.plot(dense_t, basis_kernels[basis_index](dense_t) * scale, linestyle="--", label=f"true {basis_index}")
        axis.set_xlabel("time")
        axis.legend(loc="best", fontsize=8)

    axes[0].set_ylabel("basis value")
    if show:
        plt.show()
    return axes[0].figure


def plot_estimated_intensity(
    learner,
    events=None,
    intensity_track_step=None,
    end_time=None,
    t_min: float | None = None,
    t_max: float | None = None,
    plot_nodes=None,
    node_names=None,
    n_points: int = 1000,
    show: bool = True,
    ax=None,
):
    """Plot intensities implied by a fitted Hawkes learner and event history."""

    plt = _plt()
    if events is None:
        if getattr(learner, "data", None) is None or len(learner.data) != 1:
            raise ValueError("events must be provided unless learner has exactly one fitted realization")
        events = learner.data[0]
    if end_time is None:
        if getattr(learner, "_end_times", None) is not None and len(learner._end_times):
            end_time = float(learner._end_times[0])
        else:
            end_time = max((float(ts[-1]) for ts in events if len(ts)), default=0.0)
    if intensity_track_step is None:
        intensity_track_step = max(float(end_time), 1.0) / max(int(n_points), 1)

    intensities, times = learner.estimated_intensity(events, intensity_track_step, end_time=end_time)
    times = np.asarray(times, dtype=float)
    n_nodes = len(intensities)
    nodes = list(range(n_nodes)) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(n_nodes, nodes, node_names)
    t_min = 0.0 if t_min is None else float(t_min)
    t_max = float(end_time) if t_max is None else float(t_max)
    mask = _interval_mask(times, t_min, t_max)

    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(10, max(2.5, 2.4 * len(nodes))),
        sharex=True,
        sharey=False,
    )
    if not created:
        show = False

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        axis.plot(times[mask], np.asarray(intensities[node], dtype=float)[mask], label="estimated")
        ts = np.asarray(events[node], dtype=float)
        ts = ts[_interval_mask(ts, t_min, t_max)]
        if ts.size and np.any(mask):
            y = np.interp(ts, times[mask], np.asarray(intensities[node], dtype=float)[mask])
            axis.scatter(ts, y, s=18, label="jumps")
        axis.set_ylabel(label)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("time")
    if show:
        plt.show()
    return axes[0, 0].figure


def qq_plots(
    point_process=None,
    residuals=None,
    plot_nodes=None,
    node_names=None,
    line: str = "45",
    show: bool = True,
    ax=None,
):
    """Plot exponential QQ diagnostics from compensators or residual arrays."""

    plt = _plt()
    if residuals is None:
        if point_process is None:
            raise ValueError("point_process or residuals must be provided")
        if not point_process.tracked_compensator:
            point_process.store_compensator_values()
        residuals = [np.diff(np.asarray(values, dtype=float)) for values in point_process.tracked_compensator]
    elif point_process is not None:
        raise ValueError("provide either point_process or residuals, not both")

    nodes = list(range(len(residuals))) if plot_nodes is None else list(plot_nodes)
    labels = _node_names(len(residuals), nodes, node_names)
    axes, created = _as_axes(
        ax,
        (len(nodes), 1),
        plt,
        figsize=(5.5, max(3.0, 2.8 * len(nodes))),
        sharex=True,
        sharey=True,
    )
    if not created:
        show = False

    for row, (node, label) in enumerate(zip(nodes, labels)):
        axis = axes[row, 0]
        values = np.sort(np.asarray(residuals[node], dtype=float))
        values = values[np.isfinite(values)]
        if values.size:
            probs = (np.arange(values.size) + 0.5) / values.size
            theoretical = -np.log1p(-probs)
            axis.scatter(theoretical, values, s=14)
            if line == "45":
                lim = max(float(np.max(theoretical)), float(np.max(values)))
                axis.plot([0.0, lim], [0.0, lim], color="black", linewidth=1)
        axis.set_title(label)
        axis.set_ylabel("empirical")
    axes[-1, 0].set_xlabel("theoretical exponential")
    if show:
        plt.show()
    return axes[0, 0].figure


def _extended_discrete_xaxis(x_axis, n_points=100, eps=0.10):
    x_axis = np.asarray(x_axis, dtype=float)
    min_value = float(np.min(x_axis))
    max_value = float(np.max(x_axis))
    distance = max_value - min_value
    if distance == 0:
        distance = 1.0
    return np.linspace(min_value - eps * distance, max_value + eps * distance, num=n_points)


def plot_timefunction(time_function: TimeFunction, labels=None, n_points: int = 300, show: bool = True, ax=None):
    """Plot a :class:`our_hawkes.base.TimeFunction`."""

    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        show = False

    if time_function.is_constant:
        if labels is None:
            labels = [f"value = {time_function.constant:.3g}"]
        t_values = np.arange(10, dtype=float)
        ax.plot(t_values, time_function.value(t_values), label=labels[0])
    else:
        if labels is None:
            interpolation = {
                TimeFunction.InterLinear: "linear",
                TimeFunction.InterConstLeft: "constant left",
                TimeFunction.InterConstRight: "constant right",
            }[time_function.inter_mode]
            border = {
                TimeFunction.Border0: "border zero",
                TimeFunction.BorderConstant: f"border constant {time_function.border_value:.3g}",
                TimeFunction.BorderContinue: "border continue",
                TimeFunction.Cyclic: "cyclic",
            }[time_function.border_type]
            labels = ["original points", f"{interpolation}, {border}"]
        original_t = time_function.original_t
        if time_function.border_type == TimeFunction.Cyclic:
            cycle_length = original_t[-1] - original_t[0]
            original_t = np.hstack((original_t, original_t + cycle_length, original_t + 2 * cycle_length))
        t_values = _extended_discrete_xaxis(original_t, n_points=n_points)
        ax.plot(time_function.original_t, time_function.original_y, linestyle="", marker="o", label=labels[0])
        ax.plot(t_values, time_function.value(t_values), label=labels[1])

    ax.set_xlabel("time")
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax.figure
