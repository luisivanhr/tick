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
