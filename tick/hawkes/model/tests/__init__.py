# License: BSD 3 clause

import unittest

from tick.hawkes.model.build import hawkes_model as build_model


if getattr(build_model, "__pure_python__", False):
    raise unittest.SkipTest(
        "Hawkes model tests require compiled extensions.")
