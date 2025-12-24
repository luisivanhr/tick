# License: BSD 3 clause

from __future__ import annotations


__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.solver build extension is not available; compile the C++ "
        "extensions or install a wheel with binaries."
    )


RandType_unif = 0
RandType_perm = 1

SVRG_VarianceReductionMethod_Last = 0
SVRG_VarianceReductionMethod_Average = 1
SVRG_VarianceReductionMethod_Random = 2

SVRG_StepType_Fixed = 0
SVRG_StepType_BarzilaiBorwein = 1


class _BaseSolver:
    def __init__(self, *args, **kwargs):
        _raise()


class SAGADouble(_BaseSolver):
    pass


class SAGAFloat(_BaseSolver):
    pass


class AtomicSAGADouble(_BaseSolver):
    pass


class AtomicSAGAFloat(_BaseSolver):
    pass


class SVRGDouble(_BaseSolver):
    pass


class SVRGFloat(_BaseSolver):
    pass


class MultiSVRGDouble(_BaseSolver):
    pass


class SVRGDoublePtrVector(list):
    def __init__(self, *args, **kwargs):
        _raise()


class SGDDouble(_BaseSolver):
    pass


class SGDFloat(_BaseSolver):
    pass


class SDCADouble(_BaseSolver):
    pass


class SDCAFloat(_BaseSolver):
    pass


class AdaGradDouble(_BaseSolver):
    pass


class AdaGradFloat(_BaseSolver):
    pass
