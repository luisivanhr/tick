import os
import subprocess
import sys
import unittest
from pathlib import Path


class BenchmarkSmokeTest(unittest.TestCase):
    def test_numba_hot_path_benchmark_runs_and_reports_timings(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "benchmarks" / "benchmark_numba_hot_paths.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        stdout = completed.stdout
        self.assertIn("numba_enabled=", stdout)
        self.assertIn("cold=", stdout)
        self.assertIn("warm=", stdout)
        self.assertIn("reference=", stdout)


if __name__ == "__main__":
    unittest.main()
