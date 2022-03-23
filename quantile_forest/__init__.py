import os

from ._quantile_forest import ExtraTreesQuantileRegressor
from ._quantile_forest import RandomForestQuantileRegressor

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = [
    "ExtraTreesQuantileRegressor",
    "RandomForestQuantileRegressor",
]
