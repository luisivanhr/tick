# License: BSD 3 clause

from __future__ import annotations

__pure_python__ = True


class _CompiledExtensionMissing(NotImplementedError):
    pass


def _raise() -> None:
    raise _CompiledExtensionMissing(
        "tick.base_model build extension is not available; compile the C++ "
        "extensions or install a wheel with binaries."
    )


class Model:
    def __init__(self, *args, **kwargs):
        _raise()
"""Pure-Python placeholders for base_model C++ bindings.

These stubs provide minimal interfaces consumed by the high-level model
wrappers during the rewrite. They are not performance-oriented but keep the
API surface compatible for testing.
"""

__all__ = []
