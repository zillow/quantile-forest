.. _install:

Getting Started
===============

Prerequisites
-------------

The quantile-forest package requires the following dependencies:

* python (>=3.8)
* numpy (>=1.23)
* scikit-learn (>=1.0)
* scipy (>=1.4)

Install
-------

quantile-forest can be installed using `pip`::

  pip install quantile-forest

Developer Install
-----------------

Building the package from source additionally requires the following dependencies:

* cython (>=3.0a4)

To manually build and install the package, run::

  pip install --verbose --editable .

Troubleshooting
---------------

If the build fails because SciPy is not installed, ensure OpenBLAS and LAPACK are available and accessible.

On macOS, run::

  brew install openblas
  brew install lapack
  export SYSTEM_VERSION_COMPAT=1

Test and Coverage
-----------------

To test the code::

  $ python -m pytest quantile_forest -v

To test the documentation::

  $ python -m pytest docs/*rst

Documentation
-------------

To build the documentation, run::

  $ pip install -r ./docs/sphinx_requirements.txt
  $ mkdir -p ./docs/_images
  $ sphinx-build -b html ./docs ./docs/_build

.. toctree::
   :maxdepth: 2
   :caption: Install
   :hidden:

   Getting Started <self>
