"""All minimum dependencies for quantile-forest."""

import argparse
from collections import defaultdict

CYTHON_MIN_VERSION = "3.0a4"
NUMPY_MIN_VERSION = "1.23"
SCIPY_MIN_VERSION = "1.4"
SKLEARN_MIN_VERSION = "1.5"

# 'build' and 'install' is included to have structured metadata for CI.
# The values are (version_spec, comma separated tags).
dependent_packages = {
    "cython": (CYTHON_MIN_VERSION, "build"),
    "numpy": (NUMPY_MIN_VERSION, "build, install"),
    "scipy": (SCIPY_MIN_VERSION, "build, install"),
    "scikit-learn": (SKLEARN_MIN_VERSION, "build, install"),
}

# Create inverse mapping for setuptools.
tag_to_packages: dict = defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append(f"{package}>={min_version}")

# Used by CI to get the min dependencies.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get minimum dependencies for a package.")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
