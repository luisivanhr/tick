# License: BSD 3 clause

# Templating update - 19/02/2018
#  Importing DLLs has gotten a bit strange and now requires
#  updating the path for DLLs to be found before hand

from tick.base.opsys import add_to_path_if_windows

add_to_path_if_windows(__file__)

from . import hawkes_simulation
from .hawkes_simulation import *  # noqa: F401,F403

__all__ = hawkes_simulation.__all__
