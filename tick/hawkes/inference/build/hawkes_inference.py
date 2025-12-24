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
