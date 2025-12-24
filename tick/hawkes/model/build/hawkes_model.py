# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.hawkes model build extension is not available; compile the "
        "C++ extensions or install a wheel with binaries."
    )


class _BaseModel:
    def __init__(self, *args, **kwargs):
        _raise()


class ModelHawkesExpKernLeastSq(_BaseModel):
    pass


class ModelHawkesExpKernLogLik(_BaseModel):
    pass


class ModelHawkesSumExpKernLeastSq(_BaseModel):
    pass


class ModelHawkesSumExpKernLogLik(_BaseModel):
"""Python placeholders for Hawkes model C++ bindings during rewrite."""
from __future__ import annotations


class _BaseHawkesModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def grad(self, coeffs, out):  # pragma: no cover - placeholder
        raise NotImplementedError("Hawkes models are pending rewrite")

    def loss(self, coeffs):  # pragma: no cover - placeholder
        raise NotImplementedError("Hawkes models are pending rewrite")

    def compare(self, other):
        return isinstance(other, self.__class__) and self.args == other.args and self.kwargs == other.kwargs


class ModelHawkesExpKernLeastSq(_BaseHawkesModel):
    pass


class ModelHawkesExpKernLogLik(_BaseHawkesModel):
    pass


class ModelHawkesSumExpKernLeastSq(_BaseHawkesModel):
    pass


class ModelHawkesSumExpKernLogLik(_BaseHawkesModel):
    pass
