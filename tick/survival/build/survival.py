"""Python placeholders for survival C++ bindings during rewrite."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class _BaseCox:
    def __init__(self, features, times, censoring):
        self.features = features
        self.times = times
        self.censoring = censoring.astype(np.ushort)
        self.n_samples, self.n_features = features.shape
        self.n_failures = int(self.censoring.sum())

    def get_n_coeffs(self):
        return self.n_features

    def grad(self, coeffs, out):
        # Simple partial likelihood gradient
        linear_pred = self._linear_predictor(coeffs)
        risk = np.exp(linear_pred)
        order = np.argsort(self.times)[::-1]
        cum_risk = np.cumsum(risk[order])
        cum_xrisk = np.cumsum((self.features[order].T * risk[order]).T, axis=0)

        grad = np.zeros_like(coeffs, dtype=np.result_type(coeffs, np.float64))
        for idx, ord_idx in enumerate(order):
            if not self.censoring[ord_idx]:
                continue
            denom = cum_risk[idx]
            mean_x = cum_xrisk[idx] / denom
            grad -= self.features[ord_idx]
            grad += mean_x
        out[:] = grad / max(self.n_failures, 1)

    def loss(self, coeffs):
        linear_pred = self._linear_predictor(coeffs)
        risk = np.exp(linear_pred)
        order = np.argsort(self.times)[::-1]
        cum_risk = np.cumsum(risk[order])
        loss = 0.0
        for idx, ord_idx in enumerate(order):
            if not self.censoring[ord_idx]:
                continue
            loss += -linear_pred[ord_idx] + np.log(cum_risk[idx])
        return loss / max(self.n_failures, 1)

    def _linear_predictor(self, coeffs):
        if sp.issparse(self.features):
            return self.features.dot(coeffs)
        return self.features @ coeffs

    def compare(self, other):
        if not isinstance(other, _BaseCox):
            return False
        same_meta = (
            self.n_samples == other.n_samples
            and self.n_features == other.n_features
            and self.n_failures == other.n_failures
        )
        if not same_meta:
            return False

        def _eq(arr1, arr2):
            if sp.issparse(arr1) or sp.issparse(arr2):
                return sp.issparse(arr1) and sp.issparse(arr2) and np.allclose(
                    arr1.toarray(), arr2.toarray()
                )
            return np.allclose(arr1, arr2)

        return _eq(self.features, other.features) and _eq(self.times, other.times)


class ModelCoxRegPartialLikFloat(_BaseCox):
    pass


class ModelCoxRegPartialLikDouble(ModelCoxRegPartialLikFloat):
    pass


class ModelSCCS:
    def __init__(self, features, labels, censoring, n_lags):
        self.features = features
        self.labels = labels
        self.censoring = censoring
        self.n_lags = n_lags
        self.n_cases = len(features)
        self.n_intervals, self.n_coeffs = features[0].shape

    def _softmax(self, scores):
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()

    def loss(self, coeffs):
        total = 0.0
        for x_case, y_case, cens in zip(self.features, self.labels, self.censoring):
            limit = int(cens)
            for t in range(limit):
                scores = x_case[t].reshape(-1, self.n_coeffs // (self.n_lags[0] + 1)) @ coeffs.reshape(-1, self.n_lags[0] + 1).T
                probs = self._softmax(scores.flatten())
                total -= np.log(probs[int(y_case[t])])
        return total / max(self.n_cases, 1)

    def grad(self, coeffs, out):
        grad = np.zeros_like(coeffs, dtype=np.result_type(coeffs, np.float64))
        for x_case, y_case, cens in zip(self.features, self.labels, self.censoring):
            limit = int(cens)
            for t in range(limit):
                x_t = x_case[t]
                scores = x_t.reshape(-1, self.n_coeffs // (self.n_lags[0] + 1)) @ coeffs.reshape(-1, self.n_lags[0] + 1).T
                probs = self._softmax(scores.flatten())
                grad += np.kron(np.ones(self.n_lags[0] + 1), x_t) * probs.sum()
                grad[int(y_case[t]) * (self.n_lags[0] + 1): (int(y_case[t]) + 1) * (self.n_lags[0] + 1)] -= x_t[: self.n_lags[0] + 1]
        out[:] = grad / max(self.n_cases, 1)

    def get_n_coeffs(self):
        return self.n_coeffs

    def get_epoch_size(self):
        return sum(int(c) for c in self.censoring)

    def get_rand_max(self):
        return self.get_epoch_size()

    def compare(self, other):
        if not isinstance(other, ModelSCCS):
            return False
        return (
            self.n_lags.tolist() == other.n_lags.tolist()
            and self.n_coeffs == other.n_coeffs
            and self.get_epoch_size() == other.get_epoch_size()
        )
