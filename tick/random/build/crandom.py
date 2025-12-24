# License: BSD 3 clause

from __future__ import annotations

import time
from typing import Optional

import numpy as np

__pure_python__ = True


def _rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def test_uniform(*args):
    if len(args) == 2:
        size, seed = args
        low, high = 0.0, 1.0
    elif len(args) == 4:
        low, high, size, seed = args
    elif len(args) == 3:
        low, high, size = args
        seed = None
    else:
        raise TypeError("test_uniform expects (size, seed) or (a, b, size, seed)")
    return _rng(seed).uniform(low, high, size)


def test_uniform_int(a, b, size, seed=None):
    return _rng(seed).integers(a, b, size=size)


def test_gaussian(*args):
    if len(args) == 2:
        size, seed = args
        mu, sigma = 0.0, 1.0
    elif len(args) == 4:
        mu, sigma, size, seed = args
    elif len(args) == 3:
        mu, sigma, size = args
        seed = None
    else:
        raise TypeError("test_gaussian expects (size, seed) or (mu, sigma, size, seed)")
    return _rng(seed).normal(mu, sigma, size)


def test_exponential(intensity, size, seed=None):
    scale = 1.0 / float(intensity)
    return _rng(seed).exponential(scale, size)


def test_poisson(rate, size, seed=None):
    return _rng(seed).poisson(rate, size)


def test_discrete(probabilities, size, seed=None):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    choices = np.arange(len(probabilities))
    return _rng(seed).choice(choices, size=size, p=probabilities)


def test_uniform_threaded(sample_size, wait_time=0):
    if wait_time:
        time.sleep(wait_time)
    return _rng(None).uniform(0.0, 1.0, sample_size)
