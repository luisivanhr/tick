# License: BSD 3 clause

# Templating update - 19/02/2018
#  Importing DLLs has gotten a bit strange and now requires
#  updating the path for DLLs to be found before hand

from tick.base.opsys import add_to_path_if_windows

add_to_path_if_windows(__file__)

# Pure-Python shim implementations
from . import base as base_module
from .base import *  # noqa: F401,F403

__all__ = getattr(base_module, "__all__", [])
