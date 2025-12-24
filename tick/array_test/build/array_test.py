# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True

import unittest


def _skip(*args, **kwargs):
    raise unittest.SkipTest(
        "tick.array_test build extension is not available; "
        "compile the C++ extensions or install a wheel with binaries."
    )


def test_sum_double_pointer(*args, **kwargs):
    _skip()


def test_sum_ArrayDouble(*args, **kwargs):
    _skip()


def test_sum_SArray_shared_ptr(*args, **kwargs):
    _skip()


def test_sum_VArray_shared_ptr(*args, **kwargs):
    _skip()


def __getattr__(name: str):
    return _skip
