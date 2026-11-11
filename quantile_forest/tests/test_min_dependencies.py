"""Testing for the minimum dependencies (quantile_forest._min_dependencies)."""

import tomllib
from pathlib import Path

import pytest

import quantile_forest
from quantile_forest._min_dependencies import dependent_packages, tag_to_packages

# The root of the source tree, which an installed package does not have.
ROOT = Path(quantile_forest.__file__).parent.parent


def test_min_dependencies_pyproject():
    """Check that pyproject.toml requires the minimum dependencies."""
    path = ROOT / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml is not available.")
    with open(path, "rb") as f:
        pyproject = tomllib.load(f)

    requirements = {
        "build": pyproject["build-system"]["requires"],
        "install": pyproject["project"]["dependencies"],
    }
    for tag, expected in tag_to_packages.items():
        actual = {requirement.lower() for requirement in requirements[tag]}
        assert actual - {"setuptools", "wheel"} == set(expected)


def test_min_dependencies_docs():
    """Check that the documentation lists the minimum dependencies."""
    path = ROOT / "docs" / "source" / "getting_started"
    if not path.exists():
        pytest.skip("The documentation is not available.")
    docs = "".join(
        (path / name).read_text(encoding="utf-8")
        for name in ("installation.rst", "developers.rst")
    )

    for package, (min_version, _) in dependent_packages.items():
        assert f"* {package} (>={min_version})" in docs
