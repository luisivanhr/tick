"""Print the current tick Hawkes equivalence ledger summary."""

from __future__ import annotations

import collections
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_equivalence_ledger import _discover_tick_hawkes_tests, classify_tick_test


def main() -> None:
    tests = _discover_tick_hawkes_tests()
    counts = collections.Counter(classify_tick_test(test_id)[0] for test_id in tests)
    print(f"tick Hawkes tests inventoried: {len(tests)}")
    for status in ["pass", "xfail_equivalence_gap", "skip_optional_backend", "out_of_scope_non_hawkes"]:
        print(f"{status}: {counts[status]}")


if __name__ == "__main__":
    main()
