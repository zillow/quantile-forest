###############
Getting Started
###############

Prerequisites
=============

The quantile-forest package requires the following dependencies:

* python (>=3.8)
* cython (>=3.0a4)
* scikit-learn (>=1.0)

Install
=======

quantile-forest can be installed using `pip`::

  pip install quantile-forest

Developer Install
=================

To manually build and install the package, run::

  pip install --verbose --editable .

Troubleshooting
===============

If the build fails because SciPy is not installed, ensure OpenBLAS and LAPACK are available and accessible.

On macOS, run::

  brew install openblas
  brew install lapack
  export SYSTEM_VERSION_COMPAT=1

Test and Coverage
=================

To test the code::

  $ pytest quantile_forest -v

Documentation
=============

To build the documentation, run::

  $ pip install -r ./docs/sphinx_requirements.txt
  $ sphinx-build -b html ./docs ./docs/_build
