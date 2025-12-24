# License: BSD 3 clause

import unittest

from tick.hawkes.simulation.build import hawkes_simulation as build_simulation


if getattr(build_simulation, "__pure_python__", False):
    raise unittest.SkipTest(
        "Hawkes simulation tests require compiled extensions.")
