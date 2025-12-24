# License: BSD 3 clause

import unittest

from tick.linear_model.build import linear_model as build_linear_model


if getattr(build_linear_model, "__pure_python__", False):
    raise unittest.SkipTest(
        "Linear model tests require compiled extensions.")
