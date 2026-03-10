"""
ospgrid - A plane grid elastic analysis wrapper for OpenSeesPy
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ospgrid")
except PackageNotFoundError:  # package not installed (e.g. running from source)
    __version__ = "unknown"

from .grid import *
from .utils import *
from .post import *

# The public API is controlled by __all__ in each submodule:
#   grid.py  -> Support, Node, Member, Grid
#   utils.py -> save_figs_to_file, crop_to_bbox
#   post.py  -> make_grid
