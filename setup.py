#! /usr/bin/env python
"""scikit-learn compatible quantile forests."""

import codecs
import os
import sys

from setuptools import find_packages, setup


def write_version_py():
    with open(os.path.join("quantile_forest", "version.txt")) as f:
        version = f.read().strip()

    with open(os.path.join("quantile_forest", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


__version__ = write_version_py()

with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = f.read().splitlines()

DISTNAME = "quantile-forest"
DESCRIPTION = "scikit-learn compatible quantile forests."
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Zillow Group AI Team"
LICENSE = "Apache License 2.0"
VERSION = __version__
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: Implementation :: CPython",
]


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
    )

    if len(sys.argv) == 1 or (
        len(sys.argv) >= 2
        and (
            "--help" in sys.argv[1:]
            or sys.argv[1]
            in (
                "--help-commands",
                "--version",
                "egg_info",
                "dist_info",
                "clean",
            )
        )
    ):
        # For these actions, NumPy is not required. They must succeed without
        # NumPy, for example to install the package when NumPy is not present.
        pass
    else:
        import numpy
        from Cython.Build import cythonize
        from setuptools.extension import Extension

        # For building Cython extensions.
        EXTENSIONS = [
            Extension(
                "quantile_forest._quantile_forest_fast",
                sources=["quantile_forest/_quantile_forest_fast.pyx"],
                include_dirs=[numpy.get_include()],
                language="c++",
            ),
        ]
        metadata["ext_modules"] = cythonize(EXTENSIONS)

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
