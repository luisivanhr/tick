# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.robust build extension is not available; compile the C++ "
        "extensions or install a wheel with binaries."
    )


class _BaseModel:
    def __init__(self, *args, **kwargs):
        _raise()


class ModelAbsoluteRegressionDouble(_BaseModel):
    pass


class ModelModifiedHuberDouble(_BaseModel):
    pass


class ModelHuberDouble(_BaseModel):
    pass


class ModelEpsilonInsensitiveDouble(_BaseModel):
    pass


class ModelLinRegWithInterceptsDouble(_BaseModel):
    pass
