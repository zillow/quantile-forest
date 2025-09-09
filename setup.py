#! /usr/bin/env python
"""scikit-learn compatible quantile forests."""

import builtins
import os
import sys

from setuptools import find_packages, setup

# Setting a global variable so that the main package __init__ can detect if it
# is being loaded by the setup routine.
builtins.__QF_SETUP__ = True


def write_version_py():
    with open(os.path.join("quantile_forest", "version.txt")) as f:
        version = f.read().strip()

    with open(os.path.join("quantile_forest", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')

    return version


__version__ = write_version_py()


def configure_extension_modules():
    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    if "sdist" in sys.argv or "--help" in sys.argv:
        return []

    import numpy
    from Cython.Build import cythonize
    from setuptools.extension import Extension

    # For building Cython extensions.
    EXTENSIONS = [
        Extension(
            "quantile_forest._quantile_forest_fast",
            sources=["quantile_forest/_quantile_forest_fast.pyx"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
        Extension(
            "quantile_forest._utils",
            sources=["quantile_forest/_utils.pyx"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
    ]

    return cythonize(
        EXTENSIONS,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
            "wraparound": False,
        },
    )


def setup_package():
    metadata = dict(
        packages=find_packages(),
        zip_safe=False,
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not all(command in ("egg_info", "dist_info", "clean", "check") for command in commands):
        metadata["ext_modules"] = configure_extension_modules()

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
