# License: BSD 3 clause
"""Python stand-ins for legacy C++ base helpers.

The original package relied on compiled extensions to back certain
attributes. During the pure-Python rewrite we provide minimal shims so
existing tests and attribute linkage logic can continue to operate.
"""

import errno
import os
import math
from typing import Iterable

import scipy.stats


class A0:
    """Minimal counterpart to the legacy C++ A0 class."""

    def __init__(self):
        self._cpp_int = 0

    def set_cpp_int(self, value):
        self._cpp_int = value

    def get_cpp_int(self):
        return self._cpp_int


# Exception helpers ------------------------------------------------------------

def throw_out_of_range():
    raise IndexError("out_of_range")


def throw_system_error():
    raise RuntimeError(os.strerror(errno.EACCES))


def throw_invalid_argument():
    raise ValueError("invalid_argument")


def throw_domain_error():
    raise ValueError("domain_error")


def throw_runtime_error():
    raise RuntimeError("runtime_error")


def throw_string():
    raise RuntimeError("string")


# Statistics helpers -----------------------------------------------------------

def standard_normal_cdf(x: float):
    return float(scipy.stats.norm.cdf(x))


def standard_normal_inv_cdf(x, output=None):
    values = scipy.stats.norm.ppf(x)
    if output is not None:
        output[...] = values
        return output
    return float(values) if not hasattr(values, "__len__") else values


__all__: Iterable[str] = [
    "A0",
    "throw_out_of_range",
    "throw_system_error",
    "throw_invalid_argument",
    "throw_domain_error",
    "throw_runtime_error",
    "throw_string",
    "standard_normal_cdf",
    "standard_normal_inv_cdf",
]
