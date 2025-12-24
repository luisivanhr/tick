# License: BSD 3 clause

import unittest

from tick.robust.build import robust as build_robust


if getattr(build_robust, "__pure_python__", False):
    raise unittest.SkipTest(
        "Robust model tests require compiled extensions.")
