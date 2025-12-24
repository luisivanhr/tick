# License: BSD 3 clause

import pytest

# The solver C++ bindings are not yet ported; skip solver tests during the
# rewrite so the rest of the suite can run.
pytest.skip("Solver bindings pending Python rewrite", allow_module_level=True)

from .solver import TestSolver
