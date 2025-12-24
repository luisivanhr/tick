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
"""Python placeholders for robust C++ bindings."""


import numpy as np


class _BaseRobustModel:
    def __init__(self, features, labels, fit_intercept=True, n_threads=1):
        self.features = features
        self.labels = labels
        self.fit_intercept = fit_intercept
        self.n_threads = n_threads
        self.n_samples, self.n_features = features.shape
        self.dtype = getattr(features, "dtype", np.float64)

    def set_fit_intercept(self, val):
        self.fit_intercept = bool(val)
        return self

    def _dot(self, coeffs):
        w = coeffs[:self.n_features]
        intercept = coeffs[self.n_features] if self.fit_intercept else 0.0
        return self.features.dot(w) + intercept

    def get_n_coeffs(self):
        return self.n_features + (1 if self.fit_intercept else 0)

    def compare(self, other):
        return (
            isinstance(other, self.__class__)
            and self.fit_intercept == getattr(other, "fit_intercept", None)
            and self.n_threads == getattr(other, "n_threads", None)
            and self.n_features == getattr(other, "n_features", None)
            and self.n_samples == getattr(other, "n_samples", None)
        )


class ModelLinRegWithInterceptsFloat(_BaseRobustModel):
    def __init__(self, features, labels, fit_intercept=True, n_threads=1):
        super().__init__(features, labels, fit_intercept, n_threads)
        self._n_coeffs = self.n_features + self.n_samples
        if self.fit_intercept:
            self._n_coeffs += 1

    def set_fit_intercept(self, val):
        super().set_fit_intercept(val)
        self._n_coeffs = self.n_features + self.n_samples
        if self.fit_intercept:
            self._n_coeffs += 1
        return self

    def _split_coeffs(self, coeffs):
        w = coeffs[: self.n_features]
        offset = self.n_features
        intercept = coeffs[offset] if self.fit_intercept else 0.0
        if self.fit_intercept:
            offset += 1
        sample_intercepts = coeffs[offset:]
        return w, intercept, sample_intercepts

    def loss(self, coeffs):
        w, intercept, sample_intercepts = self._split_coeffs(coeffs)
        pred = self.features.dot(w) + intercept + sample_intercepts
        residual = pred - self.labels
        return 0.5 * np.mean(residual ** 2)

    def grad(self, coeffs, out):
        w, intercept, sample_intercepts = self._split_coeffs(coeffs)
        pred = self.features.dot(w) + intercept + sample_intercepts
        residual = pred - self.labels
        scale = 1.0 / self.n_samples

        grad_w = self.features.T.dot(residual) * scale
        start = 0
        out[start : start + self.n_features] = grad_w
        start += self.n_features

        if self.fit_intercept:
            out[start] = residual.mean()
            start += 1

        out[start:] = residual * scale
class ModelLinRegWithInterceptsFloat:
    def __init__(self, n_coeffs=0, *args, **kwargs):
        self._n_coeffs = n_coeffs

    def loss(self, coeffs):  # pragma: no cover - placeholder
        return 0.0

    def grad(self, coeffs, out):  # pragma: no cover - placeholder
        out[:] = 0

    def get_n_coeffs(self):
        return self._n_coeffs

    def get_lip_mean(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 + (1.0 if self.fit_intercept else 0.0)
        return float(np.mean(row_norms + extra))

    def get_lip_max(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 + (1.0 if self.fit_intercept else 0.0)
        return float(np.max(row_norms + extra))


class ModelLinRegWithInterceptsDouble(ModelLinRegWithInterceptsFloat):
    pass


class ModelAbsoluteRegressionFloat(_BaseRobustModel):
    def loss(self, coeffs):
        pred = self._dot(coeffs)
        return np.mean(np.abs(pred - self.labels))

    def grad(self, coeffs, out):
        pred = self._dot(coeffs)
        residual = pred - self.labels
        sign = np.sign(residual)
        scale = 1.0 / self.n_samples

        out[: self.n_features] = self.features.T.dot(sign) * scale
        if self.fit_intercept:
            out[self.n_features] = sign.mean()

    def compare(self, other):
        return super().compare(other)


class ModelAbsoluteRegressionDouble(ModelAbsoluteRegressionFloat):
    pass


class _ThresholdMixin(_BaseRobustModel):
    def __init__(self, features, labels, fit_intercept=True, threshold=1.0,
                 n_threads=1):
        super().__init__(features, labels, fit_intercept, n_threads)
        self.set_threshold(threshold)

    def set_threshold(self, val):
        if val <= 0:
            raise RuntimeError("threshold must be > 0")
        self.threshold = val

    def get_threshold(self):
        return self.threshold

    def compare(self, other):
        return super().compare(other) and self.threshold == getattr(
            other, "threshold", None)


class ModelEpsilonInsensitiveFloat(_ThresholdMixin):
    def __init__(self, features, labels, fit_intercept=True, threshold=1.0,
                 n_threads=1):
        super().__init__(features, labels, fit_intercept, threshold, n_threads)

    def loss(self, coeffs):
        pred = self._dot(coeffs)
        res = np.abs(pred - self.labels)
        return np.mean(np.maximum(res - self.threshold, 0.0))

    def grad(self, coeffs, out):
        pred = self._dot(coeffs)
        diff = pred - self.labels
        mask = np.abs(diff) > self.threshold
        signed = np.sign(diff) * mask
        scale = 1.0 / self.n_samples

        out[: self.n_features] = self.features.T.dot(signed) * scale
        if self.fit_intercept:
            out[self.n_features] = signed.mean()


class ModelEpsilonInsensitiveDouble(ModelEpsilonInsensitiveFloat):
    pass


class ModelHuberFloat(_ThresholdMixin):
    def __init__(self, features, labels, fit_intercept=True, threshold=1.0,
                 n_threads=1):
        super().__init__(features, labels, fit_intercept, threshold, n_threads)

    def loss(self, coeffs):
        pred = self._dot(coeffs)
        residual = pred - self.labels
        abs_res = np.abs(residual)
        quad = abs_res <= self.threshold
        linear_part = self.threshold * (abs_res - 0.5 * self.threshold)
        return np.mean(np.where(quad, 0.5 * residual ** 2, linear_part))

    def grad(self, coeffs, out):
        pred = self._dot(coeffs)
        residual = pred - self.labels
        abs_res = np.abs(residual)
        use_res = abs_res <= self.threshold
        grad_vec = np.where(use_res, residual, self.threshold * np.sign(residual))
        scale = 1.0 / self.n_samples

        out[: self.n_features] = self.features.T.dot(grad_vec) * scale
        if self.fit_intercept:
            out[self.n_features] = grad_vec.mean()

    def get_lip_mean(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 if self.fit_intercept else 0.0
        return float(np.mean(row_norms + extra))

    def get_lip_max(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 if self.fit_intercept else 0.0
        return float(np.max(row_norms + extra))


class ModelHuberDouble(ModelHuberFloat):
    pass


class ModelModifiedHuberFloat(_BaseRobustModel):
    def loss(self, coeffs):
        pred = self._dot(coeffs)
        yz = self.labels * pred
        loss = np.where(
            yz <= -1,
            -4 * yz,
            np.where(yz < 1, (1 - yz) ** 2, 0.0),
        )
        return np.mean(loss)

    def grad(self, coeffs, out):
        pred = self._dot(coeffs)
        yz = self.labels * pred
        grad_factor = np.where(
            yz <= -1,
            -4 * self.labels,
            np.where(yz < 1, -2 * (1 - yz) * self.labels, 0.0),
        )
        scale = 1.0 / self.n_samples

        out[: self.n_features] = self.features.T.dot(grad_factor) * scale
        if self.fit_intercept:
            out[self.n_features] = grad_factor.mean()

    def compare(self, other):
        return super().compare(other)

    def get_lip_mean(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 if self.fit_intercept else 0.0
        return float(2 * np.mean(row_norms + extra))

    def get_lip_max(self):
        row_norms = np.array(self.features.multiply(self.features).sum(axis=1)).ravel() \
            if hasattr(self.features, "multiply") else np.sum(self.features ** 2, axis=1)
        extra = 1.0 if self.fit_intercept else 0.0
        return float(2 * np.max(row_norms + extra))


class ModelModifiedHuberDouble(ModelModifiedHuberFloat):
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
