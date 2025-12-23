"""Python placeholders for robust C++ bindings."""


class ModelLinRegWithInterceptsFloat:
    def __init__(self, n_coeffs=0, *args, **kwargs):
        self._n_coeffs = n_coeffs

    def loss(self, coeffs):  # pragma: no cover - placeholder
        return 0.0

    def grad(self, coeffs, out):  # pragma: no cover - placeholder
        out[:] = 0

    def get_n_coeffs(self):
        return self._n_coeffs


class ModelLinRegWithInterceptsDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelAbsoluteRegressionDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelAbsoluteRegressionFloat(ModelLinRegWithInterceptsFloat):
    pass


class ModelEpsilonInsensitiveDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelEpsilonInsensitiveFloat(ModelLinRegWithInterceptsFloat):
    pass


class ModelHuberDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelHuberFloat(ModelLinRegWithInterceptsFloat):
    pass


class ModelModifiedHuberDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelModifiedHuberFloat(ModelLinRegWithInterceptsFloat):
    pass


__all__ = [
    "ModelLinRegWithInterceptsFloat",
    "ModelLinRegWithInterceptsDouble",
    "ModelAbsoluteRegressionFloat",
    "ModelAbsoluteRegressionDouble",
    "ModelEpsilonInsensitiveFloat",
    "ModelEpsilonInsensitiveDouble",
    "ModelHuberFloat",
    "ModelHuberDouble",
    "ModelModifiedHuberFloat",
    "ModelModifiedHuberDouble",
]
