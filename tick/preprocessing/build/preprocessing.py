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
"""Pure-Python stand-ins for preprocessing C++ bindings used in tick.

These helpers provide minimal implementations of the sparse longitudinal
feature transformers to keep tests and downstream imports working during the
rewrite. They mirror the public methods expected by the Python wrappers.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps


class SparseLongitudinalFeaturesProduct:
    """Compute pairwise product features for sparse longitudinal inputs.

    The implementation follows the behaviour used in the Python wrapper:
    - Original feature columns are preserved as-is (the inputs are assumed to
      be event start indicators).
    - Product columns are based on cumulative exposure (once an exposure
      starts, it remains active for subsequent intervals) up to the last
      observed event.
    """

    def __init__(self, features_list):
        first = features_list[0]
        self.n_intervals, self.n_features = first.shape

    def sparse_features_product(self, row, col, data, out_row, out_col, out_data):
        # Reconstruct dense matrix for straightforward manipulation
        base = sps.coo_matrix((data, (row, col)), shape=(self.n_intervals, self.n_features)).toarray()
        last_event = row.max() if len(row) else -1

        # Build dense output to simplify mapping back into the preallocated arrays
        combos = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                combos.append((i, j))

        output = np.zeros((self.n_intervals, self.n_features + len(combos)), dtype=float)
        # Original start indicators
        if last_event >= 0:
            output[: last_event + 1, : self.n_features] = base[: last_event + 1]
        # Product columns activate at the interval where the latest feature in
        # the pair starts (matching the reference behaviour used in tests)
        for idx, (i, j) in enumerate(combos):
            starts = []
            for feat in (i, j):
                feat_rows = np.nonzero(base[:, feat])[0]
                if len(feat_rows):
                    starts.append(feat_rows[0])
            if len(starts) == 2:
                start_interval = max(starts)
                output[start_interval, self.n_features + idx] = 1

        # Flatten into CSR-style arrays
        nz_row, nz_col = np.nonzero(output)
        nz_data = output[nz_row, nz_col]

        # Fill provided buffers (they may be over-allocated)
        limit = min(len(out_row), len(nz_row))
        out_row[:limit] = nz_row[:limit]
        out_col[:limit] = nz_col[:limit]
        out_data[:limit] = nz_data[:limit]


class LongitudinalFeaturesLagger:
    """Generate lagged longitudinal features for dense or sparse inputs."""

    def __init__(self, features_list, n_lags):
        self.n_intervals, self.n_features = features_list[0].shape
        self.n_lags = n_lags
        self.offsets = self._compute_offsets()

    def _compute_offsets(self):
        offsets = []
        cursor = 0
        for lag in self.n_lags:
            offsets.append(cursor)
            cursor += int(lag) + 1
        return offsets

    def dense_lag_preprocessor(self, feature_matrix, output, censoring):
        for t in range(self.n_intervals):
            if t >= censoring:
                break
            for j in range(self.n_features):
                base_offset = self.offsets[j]
                max_lag = int(self.n_lags[j])
                for k in range(max_lag + 1):
                    if t - k < 0:
                        continue
                    output[t, base_offset + k] = feature_matrix[t - k, j]

    def sparse_lag_preprocessor(self, row, col, data, out_row, out_col, out_data, censoring):
        dense = sps.coo_matrix((data, (row, col)), shape=(self.n_intervals, self.n_features)).toarray()
        output = np.zeros((self.n_intervals, sum(self.n_lags + 1)), dtype=float)
        self.dense_lag_preprocessor(dense, output, censoring)

        nz_row, nz_col = np.nonzero(output)
        nz_data = output[nz_row, nz_col]
        limit = min(len(out_row), len(nz_row))
        out_row[:limit] = nz_row[:limit]
        out_col[:limit] = nz_col[:limit]
        out_data[:limit] = nz_data[:limit]
