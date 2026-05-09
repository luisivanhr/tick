# our-hawkes

Pure-Python Hawkes and point-process tools inspired by the Hawkes-related parts
of `tick`.

The package lives under `our-hawkes/`, while the import package is
`our_hawkes`.

```python
from our_hawkes.hawkes import SimuHawkesExpKernels, HawkesExpKern
```

The implementation intentionally avoids C++ extensions. It uses NumPy/SciPy for
reference numerical work, optional Numba JIT helpers for hot loops, and Python
parallelism for repeated simulations.

## Development

Use the requested environment:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m unittest discover -s tests
```

Install editable dependencies when needed:

```powershell
& "C:\Users\luisi\Documents\Programming\Python\.misc314\Scripts\python.exe" -m pip install -e ".[dev]"
```
