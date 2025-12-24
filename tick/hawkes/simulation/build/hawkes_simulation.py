# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.hawkes simulation build extension is not available; compile the "
        "C++ extensions or install a wheel with binaries."
    )


class _BaseSimu:
    def __init__(self, *args, **kwargs):
        _raise()


class Poisson(_BaseSimu):
    pass


class InhomogeneousPoisson(_BaseSimu):
    pass


class HawkesKernel(_BaseSimu):
    pass


class HawkesKernel0(_BaseSimu):
    pass


class HawkesKernelExp(_BaseSimu):
    pass


class HawkesKernelPowerLaw(_BaseSimu):
    pass


class HawkesKernelTimeFunc(_BaseSimu):
    pass


class HawkesKernelSumExp(_BaseSimu):
    pass


class Hawkes(_BaseSimu):
    pass
