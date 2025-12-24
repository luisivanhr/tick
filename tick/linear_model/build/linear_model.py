# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.linear_model build extension is not available; compile the C++ "
        "extensions or install a wheel with binaries."
    )


LinkType_identity = 0
LinkType_exponential = 1


class _BaseModel:
    def __init__(self, *args, **kwargs):
        _raise()


class ModelLinRegDouble(_BaseModel):
    pass


class ModelLinRegFloat(_BaseModel):
    pass


class ModelLogRegDouble(_BaseModel):
    pass


class ModelLogRegFloat(_BaseModel):
    pass


class ModelPoisRegDouble(_BaseModel):
    pass


class ModelPoisRegFloat(_BaseModel):
    pass


class ModelHingeDouble(_BaseModel):
    pass


class ModelHingeFloat(_BaseModel):
    pass


class ModelSmoothedHingeDouble(_BaseModel):
    pass


class ModelSmoothedHingeFloat(_BaseModel):
    pass


class ModelQuadraticHingeDouble(_BaseModel):
    pass


class ModelQuadraticHingeFloat(_BaseModel):
    pass
