# License: BSD 3 clause

import unittest

from tick.survival.build import survival as build_survival


if getattr(build_survival, "__pure_python__", False):
    raise unittest.SkipTest(
        "Survival tests require compiled extensions.")
