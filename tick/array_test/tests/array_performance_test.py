# License: BSD 3 clause
# -*- coding: utf8 -*-

import time
import unittest

import numpy as np

try:
    from tick.array_test.build.array_test import (
        test_sum_ArrayDouble,
        test_sum_SArray_shared_ptr,
        test_sum_VArray_shared_ptr,
        test_sum_double_pointer,
    )
except ImportError:  # pragma: no cover - optional C++ bindings
    test_sum_double_pointer = None
    test_sum_ArrayDouble = None
    test_sum_SArray_shared_ptr = None
    test_sum_VArray_shared_ptr = None

if test_sum_double_pointer is not None:
    ref_size = 10000
    ref_n_loops = 10000
    start = time.process_time()
    ref_result = test_sum_double_pointer(ref_size, ref_n_loops)
    end = time.process_time()
    ref_needed_time = end - start
else:  # pragma: no cover - skipped when extension missing
    ref_size = ref_n_loops = 0
    ref_result = ref_needed_time = 0.0


@unittest.skipIf(
    test_sum_double_pointer is None,
    "array_test extension is unavailable in Python rewrite",
)
class Test(unittest.TestCase):
    def test_array_speed(self):
        """Test speed of ArrayDouble matches raw double pointer performance."""
        start = time.process_time()
        result = test_sum_ArrayDouble(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.2)

    def test_sarrayptr_speed(self):
        """Test speed of SArrayDoublePtr matches raw double pointer performance."""
        start = time.process_time()
        result = test_sum_SArray_shared_ptr(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.2)

    def test_varrayptr_speed(self):
        """Test speed of VArrayDoublePtr matches raw double pointer performance."""
        start = time.process_time()
        result = test_sum_VArray_shared_ptr(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
