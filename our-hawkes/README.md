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
Known partial areas include exact tick optimizer behavior, some learner
diagnostic/objective helpers, deep tick module import paths, exact compiled
attribute semantics, and optional TensorFlow/PyTorch cumulant backends. See
[PARITY.md](PARITY.md) for the more detailed status and gaps.

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

## Development

Use the requested environment:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

The current stabilization baseline is:

- `unittest discover -s tests`: 67 tests passing, 6 skipped optional-backend or
  intentionally unsupported parity cases.
- All scripts in `examples/`: smoke-tested successfully from this checkout.
- Clean-directory import: verified with `PYTHONPATH` set to the absolute
  `our-hawkes/src` directory.

Install editable dependencies when needed:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[dev]"
```
