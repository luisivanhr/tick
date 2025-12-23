# License: BSD 3 clause
"""Pure-Python array serialization helpers.

These functions replace the legacy C++ bindings used for array IO. They
rely on NumPy and SciPy formats while keeping the same function names the
rest of the package expects.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp


def _save_npz(filepath: str, **arrays):
    """Persist arrays to a deterministic npz file.

    Using an explicit file handle prevents numpy from automatically
    appending a ``.npz`` suffix when callers supply custom extensions such
    as ``.cereal``.
    """
    with open(filepath, "wb") as f:  # pragma: no cover - trivial wrapper
        np.savez_compressed(f, **arrays)


def _load_npz(filepath: str):
    return np.load(filepath, allow_pickle=False)


def _ensure_dtype(array: np.ndarray, dtype):
    if array.dtype != dtype:
        return np.asarray(array, dtype=dtype)
    return array


def _save_dense(filepath: str, array: np.ndarray, order: str):
    array = np.asarray(array)
    if order == "F":
        array = np.array(array, order="F", copy=True)
        order_flag = "F"
    else:
        array = np.array(array, order="C", copy=True)
        order_flag = "C"
    _save_npz(filepath, kind="dense", order=order_flag, data=array)


def _load_dense(filepath: str, dtype, order: str):
    payload = _load_npz(filepath)
    data = _ensure_dtype(payload["data"], dtype)
    if order == "F":
        return np.array(data, order="F", copy=False)
    return np.array(data, order="C", copy=False)


def _save_sparse(filepath: str, matrix: sp.spmatrix, fmt: str):
    formatted = matrix.asformat(fmt)
    _save_npz(
        filepath,
        kind="sparse",
        format=fmt,
        data=formatted.data,
        indices=formatted.indices,
        indptr=formatted.indptr,
        shape=formatted.shape,
    )


def _load_sparse(filepath: str, dtype, fmt: str):
    payload = _load_npz(filepath)
    data = _ensure_dtype(payload["data"], dtype)
    indices = payload["indices"]
    indptr = payload["indptr"]
    shape = tuple(payload["shape"])
    constructor = sp.csr_matrix if fmt == "csr" else sp.csc_matrix
    return constructor((data, indices, indptr), shape=shape)


# Dense arrays -----------------------------------------------------------------

def tick_float_array_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float32), order="C")


def tick_double_array_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float64), order="C")


def tick_float_array2d_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float32), order="C")


def tick_double_array2d_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float64), order="C")


def tick_float_colmaj_array2d_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float32), order="F")


def tick_double_colmaj_array2d_to_file(filepath: str, array: np.ndarray):
    _save_dense(filepath, _ensure_dtype(array, np.float64), order="F")


# Sparse arrays ----------------------------------------------------------------

def tick_float_sparse2d_to_file(filepath: str, matrix: sp.spmatrix):
    _save_sparse(filepath, matrix.astype(np.float32), fmt="csr")


def tick_double_sparse2d_to_file(filepath: str, matrix: sp.spmatrix):
    _save_sparse(filepath, matrix.astype(np.float64), fmt="csr")


def tick_float_colmaj_sparse2d_to_file(filepath: str, matrix: sp.spmatrix):
    _save_sparse(filepath, matrix.astype(np.float32), fmt="csc")


def tick_double_colmaj_sparse2d_to_file(filepath: str, matrix: sp.spmatrix):
    _save_sparse(filepath, matrix.astype(np.float64), fmt="csc")


# Dense loaders ----------------------------------------------------------------

def tick_float_array_from_file(filepath: str):
    return _load_dense(filepath, np.float32, order="C")


def tick_double_array_from_file(filepath: str):
    return _load_dense(filepath, np.float64, order="C")


def tick_float_array2d_from_file(filepath: str):
    return _load_dense(filepath, np.float32, order="C")


def tick_double_array2d_from_file(filepath: str):
    return _load_dense(filepath, np.float64, order="C")


def tick_float_colmaj_array2d_from_file(filepath: str):
    return _load_dense(filepath, np.float32, order="F")


def tick_double_colmaj_array2d_from_file(filepath: str):
    return _load_dense(filepath, np.float64, order="F")


# Sparse loaders ---------------------------------------------------------------

def tick_float_sparse2d_from_file(filepath: str):
    return _load_sparse(filepath, np.float32, fmt="csr")


def tick_double_sparse2d_from_file(filepath: str):
    return _load_sparse(filepath, np.float64, fmt="csr")


def tick_float_colmaj_sparse2d_from_file(filepath: str):
    return _load_sparse(filepath, np.float32, fmt="csc")


def tick_double_colmaj_sparse2d_from_file(filepath: str):
    return _load_sparse(filepath, np.float64, fmt="csc")


__all__: Iterable[str] = [
    name
    for name in list(globals())
    if name.startswith("tick_") or name.endswith("_from_file") or name.endswith("_to_file")
]
