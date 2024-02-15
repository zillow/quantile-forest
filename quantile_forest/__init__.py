"""`quantile_forest` module that implements quantile regression forests."""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

try:
    # This variable is injected in the __builtins__ by the build process. It
    # is used to enable importing subpackages of quantile-forest when the
    # binaries are not built.
    __QF_SETUP__  # type: ignore
except NameError:
    __QF_SETUP__ = False

if __QF_SETUP__:
    sys.stderr.write("Partial import of quantile-forest during the build process.\n")
    # We are not importing the rest of quantile-forest during the build
    # process, as it may not be compiled yet
else:
    from ._quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor

    __all__ = [
        "ExtraTreesQuantileRegressor",
        "RandomForestQuantileRegressor",
    ]
