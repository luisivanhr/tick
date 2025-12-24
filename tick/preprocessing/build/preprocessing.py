# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np


class LongitudinalFeaturesLagger:
    def __init__(self, features: Iterable, n_lags: np.ndarray):
        self.n_lags = np.asarray(n_lags, dtype="uint64")
        self.n_features = int(self.n_lags.shape[0])
        self.n_output_features = int((self.n_lags + 1).sum())

    def _feature_offsets(self) -> np.ndarray:
        offsets = np.zeros(self.n_features, dtype=int)
        running = 0
        for idx, lag in enumerate(self.n_lags):
            offsets[idx] = running
            running += int(lag) + 1
        return offsets

    def dense_lag_preprocessor(self, feature_matrix: np.ndarray,
                               output: np.ndarray, censoring_i: int) -> None:
        offsets = self._feature_offsets()
        n_intervals = feature_matrix.shape[0]
        limit = min(censoring_i, n_intervals)
        for feat_idx in range(self.n_features):
            lag_count = int(self.n_lags[feat_idx])
            col_offset = offsets[feat_idx]
            for lag in range(lag_count + 1):
                dest_col = col_offset + lag
                src = feature_matrix[:limit - lag, feat_idx]
                output[lag:limit, dest_col] = src

    def sparse_lag_preprocessor(self, row: np.ndarray, col: np.ndarray,
                                data: np.ndarray, out_row: np.ndarray,
                                out_col: np.ndarray, out_data: np.ndarray,
                                censoring_i: int) -> None:
        offsets = self._feature_offsets()
        entries_row: List[int] = []
        entries_col: List[int] = []
        entries_data: List[float] = []
        for r, c, v in zip(row, col, data):
            lag_count = int(self.n_lags[c])
            col_offset = offsets[c]
            for lag in range(lag_count + 1):
                dest_row = int(r) + lag
                if dest_row >= censoring_i:
                    break
                entries_row.append(dest_row)
                entries_col.append(col_offset + lag)
                entries_data.append(float(v))
        nnz = min(len(entries_row), len(out_row))
        out_row[:nnz] = entries_row[:nnz]
        out_col[:nnz] = entries_col[:nnz]
        out_data[:nnz] = entries_data[:nnz]


class SparseLongitudinalFeaturesProduct:
    def __init__(self, features: Iterable):
        first = next(iter(features))
        self.n_features = int(first.shape[1])
        self.pairs = list(combinations(range(self.n_features), 2))
        self.pair_index: Dict[Tuple[int, int], int] = {
            pair: idx for idx, pair in enumerate(self.pairs)
        }

    def sparse_features_product(self, row: np.ndarray, col: np.ndarray,
                                data: np.ndarray, out_row: np.ndarray,
                                out_col: np.ndarray, out_data: np.ndarray) -> None:
        rows: Dict[int, List[Tuple[int, float]]] = {}
        for r, c, v in zip(row, col, data):
            rows.setdefault(int(r), []).append((int(c), float(v)))

        entries_row: List[int] = []
        entries_col: List[int] = []
        entries_data: List[float] = []

        for r, entries in rows.items():
            for c, v in entries:
                entries_row.append(r)
                entries_col.append(c)
                entries_data.append(v)
            for i, (c1, v1) in enumerate(entries):
                for c2, v2 in entries[i + 1:]:
                    pair = (min(c1, c2), max(c1, c2))
                    pair_idx = self.pair_index[pair]
                    entries_row.append(r)
                    entries_col.append(self.n_features + pair_idx)
                    entries_data.append(v1 * v2)

        nnz = min(len(entries_row), len(out_row))
        out_row[:nnz] = entries_row[:nnz]
        out_col[:nnz] = entries_col[:nnz]
        out_data[:nnz] = entries_data[:nnz]
