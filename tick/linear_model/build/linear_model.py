"""Python fallbacks for linear_model C++ bindings."""
from __future__ import annotations

import numpy as np


def _prepare_features_and_intercept(features, coeffs, fit_intercept):
    if fit_intercept:
        weights = coeffs[:-1]
        intercept = coeffs[-1]
    else:
        weights = coeffs
        intercept = 0.0
    linear_term = features.dot(weights) + intercept
    return linear_term, weights


class _BaseGLM:
    def __init__(self, features, labels, fit_intercept=True, n_threads=1):
        self.features = features
        self.labels = labels
        self.fit_intercept = fit_intercept
        self.n_threads = n_threads
        self.n_samples, self.n_features = features.shape

    def get_n_coeffs(self):
        return self.n_features + (1 if self.fit_intercept else 0)

    @property
    def dtype(self):
        return self.features.dtype


class ModelLinRegFloat(_BaseGLM):
    def loss(self, coeffs):
        pred, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        resid = pred - self.labels
        return 0.5 * np.mean(resid ** 2)

    def grad(self, coeffs, out):
        pred, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        resid = pred - self.labels
        grad_w = self.features.T.dot(resid) / self.n_samples
        if self.fit_intercept:
            out[:-1] = grad_w
            out[-1] = resid.mean()
        else:
            out[:] = grad_w

    def get_lip_max(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return norms.max() / self.n_samples

    def get_lip_mean(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return norms.mean() / self.n_samples


class ModelLinRegDouble(ModelLinRegFloat):
    pass


class ModelLogRegFloat(_BaseGLM):
    def loss(self, coeffs):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        y = self.labels
        z = y * linear_term
        return np.logaddexp(0, -z).mean()

    def grad(self, coeffs, out):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        y = self.labels
        prob = -y / (1.0 + np.exp(z := y * linear_term))
        grad_w = self.features.T.dot(prob) / self.n_samples
        if self.fit_intercept:
            out[:-1] = grad_w
            out[-1] = prob.mean()
        else:
            out[:] = grad_w

    def get_lip_best(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return 0.25 * norms.max() / self.n_samples

    def get_lip_max(self):
        return self.get_lip_best()

    def get_lip_mean(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return 0.25 * norms.mean() / self.n_samples


class ModelLogRegDouble(ModelLogRegFloat):
    pass


class ModelPoisRegFloat(_BaseGLM):
    def __init__(self, features, labels, fit_intercept=True, n_threads=1,
                 link="exponential"):
        super().__init__(features, labels, fit_intercept, n_threads)
        self.link = link

    def loss(self, coeffs):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        mu = np.exp(linear_term)
        return np.mean(mu - self.labels * linear_term)

    def grad(self, coeffs, out):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        mu = np.exp(linear_term)
        residual = mu - self.labels
        grad_w = self.features.T.dot(residual) / self.n_samples
        if self.fit_intercept:
            out[:-1] = grad_w
            out[-1] = residual.mean()
        else:
            out[:] = grad_w

    def sdca_primal_dual_relation(self, l_l2sq, dual_vector, primal_out):
        # Simple relation consistent with identity link SDCA expectations
        primal_out[:] = dual_vector

    def get_lip_max(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return norms.max() * np.exp(np.clip(self.features, -20, 20)).mean() / self.n_samples

    def get_lip_mean(self):
        norms = np.linalg.norm(self.features, axis=1) ** 2
        return norms.mean() / self.n_samples


class ModelPoisRegDouble(ModelPoisRegFloat):
    pass


# Additional placeholder models required for imports
class ModelHingeFloat(_BaseGLM):
    def loss(self, coeffs):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        margin = 1 - self.labels * linear_term
        return np.maximum(0, margin).mean()

    def grad(self, coeffs, out):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        mask = (1 - self.labels * linear_term) > 0
        scaled = -self.labels * mask.astype(self.dtype)
        grad_w = self.features.T.dot(scaled) / self.n_samples
        if self.fit_intercept:
            out[:-1] = grad_w
            out[-1] = scaled.mean()
        else:
            out[:] = grad_w


class ModelHingeDouble(ModelHingeFloat):
    pass


class ModelQuadraticHingeFloat(ModelHingeFloat):
    def loss(self, coeffs):
        linear_term, _ = _prepare_features_and_intercept(self.features, coeffs, self.fit_intercept)
        margin = 1 - self.labels * linear_term
        margin = np.maximum(0, margin)
        return 0.5 * np.mean(margin ** 2)


class ModelQuadraticHingeDouble(ModelQuadraticHingeFloat):
    pass


class ModelSmoothedHingeFloat(ModelHingeFloat):
    pass


class ModelSmoothedHingeDouble(ModelSmoothedHingeFloat):
    pass


# Link type constants expected by Poisson wrappers
LinkType_identity = "identity"
LinkType_exponential = "exponential"

__all__ = [
    "LinkType_identity",
    "LinkType_exponential",
    "ModelLinRegFloat",
    "ModelLinRegDouble",
    "ModelLogRegFloat",
    "ModelLogRegDouble",
    "ModelPoisRegFloat",
    "ModelPoisRegDouble",
]
