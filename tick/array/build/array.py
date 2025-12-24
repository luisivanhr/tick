# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True

from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse


def _as_path(path: Union[str, Path]) -> Path:
    return Path(path)


def _save_dense(path: Union[str, Path], array: np.ndarray) -> None:
    path = _as_path(path)
    with path.open("wb") as handle:
        np.save(handle, array)


def _load_dense(path: Union[str, Path]) -> np.ndarray:
    path = _as_path(path)
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def _save_sparse(path: Union[str, Path], array: sparse.spmatrix) -> None:
    path = _as_path(path)
    with path.open("wb") as handle:
        sparse.save_npz(handle, array, compressed=False)


def _load_sparse(path: Union[str, Path]) -> sparse.spmatrix:
    path = _as_path(path)
    with path.open("rb") as handle:
        return sparse.load_npz(handle)


def tick_float_array_to_file(path: Union[str, Path], array: np.ndarray) -> None:
    _save_dense(path, np.asarray(array, dtype=np.float32))


def tick_double_array_to_file(path: Union[str, Path], array: np.ndarray) -> None:
    _save_dense(path, np.asarray(array, dtype=np.float64))


def tick_float_array2d_to_file(path: Union[str, Path], array: np.ndarray) -> None:
    _save_dense(path, np.asarray(array, dtype=np.float32))


def tick_double_array2d_to_file(path: Union[str, Path], array: np.ndarray) -> None:
    _save_dense(path, np.asarray(array, dtype=np.float64))


def tick_float_colmaj_array2d_to_file(
    path: Union[str, Path], array: np.ndarray
) -> None:
    array = np.asarray(array, dtype=np.float32, order="F")
    _save_dense(path, array)


def tick_double_colmaj_array2d_to_file(
    path: Union[str, Path], array: np.ndarray
) -> None:
    array = np.asarray(array, dtype=np.float64, order="F")
    _save_dense(path, array)


def tick_float_sparse2d_to_file(
    path: Union[str, Path], array: sparse.spmatrix
) -> None:
    _save_sparse(path, array.astype(np.float32))


def tick_double_sparse2d_to_file(
    path: Union[str, Path], array: sparse.spmatrix
) -> None:
    _save_sparse(path, array.astype(np.float64))


def tick_float_colmaj_sparse2d_to_file(
    path: Union[str, Path], array: sparse.spmatrix
) -> None:
    _save_sparse(path, sparse.csc_matrix(array, dtype=np.float32))


def tick_double_colmaj_sparse2d_to_file(
    path: Union[str, Path], array: sparse.spmatrix
) -> None:
    _save_sparse(path, sparse.csc_matrix(array, dtype=np.float64))


def tick_float_array_from_file(path: Union[str, Path]) -> np.ndarray:
    return _load_dense(path).astype(np.float32)


def tick_double_array_from_file(path: Union[str, Path]) -> np.ndarray:
    return _load_dense(path).astype(np.float64)


def tick_float_array2d_from_file(path: Union[str, Path]) -> np.ndarray:
    return _load_dense(path).astype(np.float32)


def tick_double_array2d_from_file(path: Union[str, Path]) -> np.ndarray:
    return _load_dense(path).astype(np.float64)


def tick_float_colmaj_array2d_from_file(path: Union[str, Path]) -> np.ndarray:
    return np.asarray(_load_dense(path), dtype=np.float32, order="F")


def tick_double_colmaj_array2d_from_file(path: Union[str, Path]) -> np.ndarray:
    return np.asarray(_load_dense(path), dtype=np.float64, order="F")


def tick_float_sparse2d_from_file(path: Union[str, Path]) -> sparse.spmatrix:
    return _load_sparse(path).astype(np.float32)


def tick_double_sparse2d_from_file(path: Union[str, Path]) -> sparse.spmatrix:
    return _load_sparse(path).astype(np.float64)


def tick_float_colmaj_sparse2d_from_file(
    path: Union[str, Path]
) -> sparse.spmatrix:
    return sparse.csc_matrix(_load_sparse(path), dtype=np.float32)


def tick_double_colmaj_sparse2d_from_file(
    path: Union[str, Path]
) -> sparse.spmatrix:
    return sparse.csc_matrix(_load_sparse(path), dtype=np.float64)
