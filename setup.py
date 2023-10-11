#! /usr/bin/env python
"""scikit-learn compatible quantile forests."""

import builtins
import codecs
import os
import sys

from setuptools import find_packages, setup

# Setting a global variable so that the main package __init__ can detect if it
# is being loaded by the setup routine.
builtins.__QF_SETUP__ = True

import quantile_forest._min_dependencies as min_deps  # noqa


def write_version_py():
    with open(os.path.join("quantile_forest", "version.txt")) as f:
        version = f.read().strip()

    with open(os.path.join("quantile_forest", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')

    return version


__version__ = write_version_py()

DISTNAME = "quantile-forest"
DESCRIPTION = "scikit-learn compatible quantile forests."
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Zillow Group AI Team"
LICENSE = "Apache License 2.0"
URL = "https://zillow.github.io/quantile-forest"
DOWNLOAD_URL = "https://pypi.org/project/quantile-forest/#files"
PROJECT_URLS = {
    "Documentation": "https://zillow.github.io/quantile-forest",
    "Source": "https://github.com/zillow/quantile-forest",
    "Tracker": "https://github.com/zillow/quantile-forest/issues",
}
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
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: Implementation :: CPython",
]


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
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
    ]

    return cythonize(EXTENSIONS, compiler_directives={"language_level": "3"})


def setup_package():
    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        install_requires=min_deps.tag_to_packages["install"],
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not all(command in ("egg_info", "dist_info", "clean", "check") for command in commands):
        metadata["ext_modules"] = configure_extension_modules()

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
