# License: BSD 3 clause

import unittest

from tick.hawkes.inference.build import hawkes_inference as build_inference


if getattr(build_inference, "__pure_python__", False):
    raise unittest.SkipTest(
        "Hawkes inference tests require compiled extensions.")
