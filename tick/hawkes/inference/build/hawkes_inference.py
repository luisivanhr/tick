# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.hawkes inference build extension is not available; compile the "
        "C++ extensions or install a wheel with binaries."
    )


class _BaseInference:
    def __init__(self, *args, **kwargs):
        _raise()


class HawkesADM4(_BaseInference):
    pass


class HawkesBasisKernels(_BaseInference):
    pass


class HawkesSumGaussians(_BaseInference):
    pass


class PointProcessCondLaw(_BaseInference):
    pass


class HawkesCumulant(_BaseInference):
    pass


class HawkesTheoreticalCumulant(_BaseInference):
    pass


class HawkesEM(_BaseInference):
    pass
"""Placeholder Hawkes inference backends for the Python rewrite."""
from __future__ import annotations

import numpy as np

from tick.base import Base


class _BaseHawkesInference(Base):
    _attrinfos = {"_n_nodes": {"writable": True}}

    def __init__(self, n_nodes: int | None = None, n_threads: int = 1):
        super().__init__()
        self._n_nodes = n_nodes
        self._n_threads = n_threads
        self._objective = 0.0

    # Generic getters/setters -----------------------------------------------------------
    def set_n_threads(self, n_threads: int):
        self._n_threads = n_threads

    def get_n_threads(self) -> int:
        return self._n_threads

    def get_obj(self) -> float:
        return float(self._objective)

    def fit(self, *args, **kwargs):
        return self

    def solve(self, *args, **kwargs):
        return self


class HawkesADM4(_BaseHawkesInference):
    def __init__(self, n_nodes: int, decay: float, n_threads: int = 1):
        super().__init__(n_nodes=n_nodes, n_threads=n_threads)
        self._decay = decay
        self._baseline = np.zeros(n_nodes)
        self._adjacency = np.zeros((n_nodes, n_nodes))

    def set_max_iter(self, *_):
        return self

    def set_c(self, *_):
        return self

    def set_decay(self, decay: float):
        self._decay = decay

    def get_adjacency(self):
        return self._adjacency

    def get_baseline(self):
        return self._baseline


class HawkesExpKern(_BaseHawkesInference):
    def __init__(self, decay: float, n_threads: int = 1, **_):
        super().__init__(n_threads=n_threads)
        self._decay = decay
        self._baseline = None
        self._adjacency = None

    def get_baseline(self):
        return self._baseline

    def get_adjacency(self):
        return self._adjacency


class HawkesSumExpKern(HawkesExpKern):
    def __init__(self, decays, n_threads: int = 1, **_):
        super().__init__(decay=float(decays[0]) if len(decays) else 0.0, n_threads=n_threads)
        self._decays = np.asarray(decays, dtype=float)

    def get_kernel_norms(self, *_, **__):
        return np.zeros_like(self._decays)


class HawkesBasisKernels(_BaseHawkesInference):
    def __init__(self, kernel_support: float, kernel_size: int, n_threads: int = 1):
        super().__init__(n_threads=n_threads)
        self._kernel_support = kernel_support
        self._kernel_size = kernel_size

    def get_kernel_support(self):
        return self._kernel_support

    def set_kernel_support(self, value: float):
        self._kernel_support = value

    def get_kernel_size(self):
        return self._kernel_size

    def set_kernel_size(self, value: int):
        self._kernel_size = value


class HawkesConditionalLaw(_BaseHawkesInference):
    def __init__(self, **_):
        super().__init__()
        self.counting: list[np.ndarray] | None = None
        self.decays: list[float] | None = None

    def set_max_nbr_intervals(self, *_):
        return self

    def set_intensity_track_step(self, *_):
        return self

    def get_mean_intensity(self):
        return 0.0

    def get_lagging_times(self):
        return np.array([])

    def get_estimated_kernels_time_func(self):
        return []


class PointProcessCondLaw(HawkesConditionalLaw):
    """Alias placeholder kept for compatibility with historical bindings."""


class HawkesEM(_BaseHawkesInference):
    def __init__(self, kernel_support: float | None = None, kernel_size: int | None = None, n_threads: int = 1):
        super().__init__(n_threads=n_threads)
        self._kernel_support = kernel_support
        self._kernel_size = kernel_size
        self._kernel_discretization = None if kernel_support is None or kernel_size is None else kernel_support / kernel_size

    def get_kernel_norms(self, *_):
        return np.array([])

    def get_kernel_primitives(self, *_):
        return np.array([])

    def get_kernel_support(self):
        return self._kernel_support

    def set_kernel_support(self, value: float):
        self._kernel_support = value

    def get_kernel_size(self):
        return self._kernel_size

    def set_kernel_size(self, value: int):
        self._kernel_size = value

    def get_kernel_fixed_dt(self):
        return False

    def set_kernel_dt(self, dt: float):
        self._kernel_discretization = dt

    def get_kernel_discretization(self):
        return self._kernel_discretization

    def loglikelihood(self, *_):
        return 0.0

    def set_buffer_variables_for_integral_of_intensity(self, *_):
        return self

    def primitive_of_intensity_at_jump_times(self, *_):
        return np.array([])


class HawkesSumGaussians(_BaseHawkesInference):
    def __init__(self, n_gaussians: int, max_mean_gaussian: float, min_mean_gaussian: float, std_gaussian: float,
                 n_threads: int = 1):
        super().__init__(n_threads=n_threads)
        self._n_gaussians = n_gaussians
        self._means = np.linspace(min_mean_gaussian, max_mean_gaussian, n_gaussians)
        self._std = std_gaussian

    def get_kernel_support(self):
        return float(self._means[-1] + 5 * self._std) if len(self._means) else 0.0

    def get_kernel_values(self, *_):
        return np.array([])


class HawkesCumulant(_BaseHawkesInference):
    def __init__(self, **_):
        super().__init__()

    def get_adjacency(self):
        return None


class HawkesTheoreticalCumulant(_BaseHawkesInference):
    def __init__(self, **_):
        super().__init__()

    def get_covariance(self):
        return None


__all__ = [
    "HawkesADM4",
    "HawkesExpKern",
    "HawkesSumExpKern",
    "HawkesBasisKernels",
    "HawkesConditionalLaw",
    "HawkesEM",
    "HawkesSumGaussians",
    "HawkesCumulant",
    "HawkesTheoreticalCumulant",
    "PointProcessCondLaw",
]
