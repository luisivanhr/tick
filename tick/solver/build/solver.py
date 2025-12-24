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
"""Python placeholders for solver C++ bindings.

These lightweight classes mimic the attribute plumbing expected by the Python
solver wrappers so parameter setters/getters work during the rewrite, even
though optimization is not yet delegated to native implementations.
"""

from __future__ import annotations

import numpy as np
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
        self._record_every = 1
        self._step = 0.0
        self._starting_iterate = None
        self._prox = None
        self._model = None
        self._prev_obj = None

    # --- basic hooks used by wrappers ---
    def set_model(self, model, *args, **kwargs):
        self._model = model
        return self

    def set_prox(self, prox, *args, **kwargs):
        self._prox = prox
        return self

    def set_starting_iterate(self, iterate):
        self._starting_iterate = None if iterate is None else np.array(iterate, copy=True)

    def set_step(self, step):
        self._step = step

    def get_step(self):
        return self._step

    def set_record_every(self, record_every):
        self._record_every = int(record_every)

    def get_record_every(self):
        return self._record_every

    def set_prev_obj(self, prev_obj):
        self._prev_obj = prev_obj

    # --- API placeholders required by specific solvers ---
    def set_variance_reduction(self, *args, **kwargs):
        return self

    def set_step_type(self, *args, **kwargs):
        return self

    # --- minimal solve/serialization helpers ---
    def solve(self, *args, **kwargs):
        # Simply keep any starting iterate as the minimizer placeholder
        return self

    def get_minimizer(self, out):
        if self._starting_iterate is None:
            out[:] = 0
        else:
            out[:] = self._starting_iterate
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
    def __init__(self, *args, **kwargs):
        _raise()


class SGDDouble(_BaseSolver):
    pass


class SGDFloat(_BaseSolver):
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
