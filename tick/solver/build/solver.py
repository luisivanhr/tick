"""Python placeholders for solver C++ bindings."""

# Randomization types used by stochastic solvers
RandType_perm = "perm"
RandType_unif = "unif"
RandType_rand = "rand"

SVRG_VarianceReductionMethod_Last = "last"
SVRG_VarianceReductionMethod_Average = "average"
SVRG_VarianceReductionMethod_Random = "random"
SVRG_StepType_Fixed = "fixed"
SVRG_StepType_BarzilaiBorwein = "bb"


class _BaseSolver:
    def __init__(self, *args, **kwargs):
        pass

    def set_model(self, *args, **kwargs):
        return self

    def set_prox(self, *args, **kwargs):
        return self

    def solve(self, *args, **kwargs):
        return None


class SGDDouble(_BaseSolver):
    pass


class SGDFloat(_BaseSolver):
    pass


class SVRGDouble(_BaseSolver):
    pass


class SVRGFloat(_BaseSolver):
    pass


class MultiSVRGDouble(_BaseSolver):
    pass


class SVRGDoublePtrVector(list):
    pass


class SAGADouble(_BaseSolver):
    pass


class SAGAFloat(_BaseSolver):
    pass


class AtomicSAGADouble(_BaseSolver):
    pass


class AtomicSAGAFloat(_BaseSolver):
    pass


class SDCADouble(_BaseSolver):
    pass


class SDCAFloat(_BaseSolver):
    pass


class AdaGradDouble(_BaseSolver):
    pass


class AdaGradFloat(_BaseSolver):
    pass


__all__ = [
    "RandType_perm",
    "RandType_unif",
    "RandType_rand",
    "SVRG_VarianceReductionMethod_Last",
    "SVRG_VarianceReductionMethod_Average",
    "SVRG_VarianceReductionMethod_Random",
    "SVRG_StepType_Fixed",
    "SVRG_StepType_BarzilaiBorwein",
    "SGDDouble",
    "SGDFloat",
    "SVRGDouble",
    "SVRGFloat",
    "MultiSVRGDouble",
    "SVRGDoublePtrVector",
    "SAGADouble",
    "SAGAFloat",
    "AtomicSAGADouble",
    "AtomicSAGAFloat",
    "SDCADouble",
    "SDCAFloat",
    "AdaGradDouble",
    "AdaGradFloat",
]
