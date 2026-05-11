# our-hawkes

Pure-Python Hawkes and point-process tools inspired by the Hawkes-related parts
of `tick`.

The package lives under `our-hawkes/`, while the import package is
`our_hawkes`.

```python
from our_hawkes.hawkes import SimuHawkesExpKernels, HawkesExpKern
```

See [PARITY.md](PARITY.md) for the current short tick Hawkes API parity matrix.

The implementation intentionally avoids C++ extensions. It uses NumPy/SciPy for
reference numerical work, optional Numba JIT helpers for hot loops, and Python
parallelism for repeated simulations.

## Optional Numba Acceleration

Numba is optional. The Python/NumPy formulas remain the source of truth and the
package imports without Numba installed. Install the extra when you want JIT
dispatchers:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[numba]"
```

Disable JIT dispatch in a process with:

```powershell
$env:OUR_HAWKES_DISABLE_NUMBA = "1"
```

The first call to a JIT-backed helper includes compile latency. Benchmark cold,
warm, and reference timings separately:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" benchmarks\benchmark_numba_hot_paths.py
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests -p test_benchmark_smoke.py
```

Numba cache files (`.nbc` / `.nbi`) may be written near `__pycache__`; on
Windows, path or permission issues should be handled by disabling Numba rather
than changing numerical behavior.

## Current API status

`our_hawkes.hawkes` exports the Hawkes-focused public names from tick's Hawkes
surface:

- Kernels: `HawkesKernel0`, `HawkesKernelExp`, `HawkesKernelSumExp`,
  `HawkesKernelPowerLaw`, and `HawkesKernelTimeFunc`.
- Simulation: `SimuPoissonProcess`, `SimuInhomogeneousPoisson`, `SimuHawkes`,
  `SimuHawkesExpKernels`, `SimuHawkesSumExpKernels`, and `SimuHawkesMulti`.
- Parametric models and learners: exponential and sum-exponential log-likelihood
  and least-squares models, `HawkesExpKern`, `HawkesSumExpKern`, and `HawkesADM4`.
- Non-parametric and cumulant learners: `HawkesEM`, `HawkesBasisKernels`,
  `HawkesSumGaussians`, `HawkesConditionalLaw`, `HawkesCumulantMatching`,
  `HawkesCumulantMatchingPyT`, and `HawkesCumulantMatchingTf`.
- Hawkes plotting helpers are available from `our_hawkes.plot` and
  `our_hawkes.hawkes.plot`.

This is not full tick parity. The current implementation is intended to be
usable for Hawkes and point-process workflows while remaining pure Python.
Known partial areas include exact C++ RNG stream parity, deep tick module import
paths, exact compiled attribute semantics, and optional TensorFlow/PyTorch
cumulant solve backends. See
[PARITY.md](PARITY.md) for the more detailed test-backed status and gaps.

## Examples

The scripts in `examples/` prepend the local `src/` directory to `sys.path`, so
they run against this checkout without installing `our-hawkes` and without
depending on `tick`.

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" examples\plot_hawkes_simulation.py
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" examples\plot_hawkes_em.py
```

All example scripts are intended to run headlessly with Matplotlib's `Agg`
backend.

The notebook `examples/hawkes_time_rescaling_gof.ipynb` demonstrates
time-rescaling goodness-of-fit diagnostics for univariate Hawkes processes with
exponential, sum-exponential, power-law, and time-function kernels.

## Development

Use the requested environment:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

The current stabilization baseline is:

- `unittest discover -s tests`: 135 tests run, OK, with 1 skipped documented
  parity case.
- `unittest discover -s tests\tick_equivalence`: 171 local tick Hawkes tests
  inventoried in the equivalence ledger; current classification is 165 pass,
  0 unresolved equivalence gaps, and 6 optional backend cases.
- Tick-equivalence status: 165 pass, 0 xfail, 6 optional skips.
- `tests\tick_equivalence\report_equivalence.py`: prints the ledger counts
  by `pass`, `xfail_equivalence_gap`, and `skip_optional_backend`.
- All scripts in `examples/`: smoke-tested successfully from this checkout.
- Clean-directory import: verified with `PYTHONPATH` set to the absolute
  `our-hawkes/src` directory.
- Public export check: every name listed in `our_hawkes.hawkes.__all__` imports
  from the local source tree.

Install editable dependencies when needed:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[dev]"
```
