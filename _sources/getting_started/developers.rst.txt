.. _developers:

Developer's Guide
-----------------

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Building the package from source additionally requires the following dependencies:

* cython (>=3.0a4)

To manually build and install the package, run::

  pip install --verbose --editable .

Troubleshooting
~~~~~~~~~~~~~~~

If the build fails because SciPy is not installed, ensure OpenBLAS and LAPACK are available and accessible.

On macOS, run::

  brew install openblas
  brew install lapack
  export SYSTEM_VERSION_COMPAT=1

Test and Coverage
~~~~~~~~~~~~~~~~~

To test the code::

  $ python -m pytest quantile_forest -v

To test the documentation::

  $ python -m pytest docs/user_guide/*rst

Documentation
~~~~~~~~~~~~~

To build the documentation, run::

  $ pip install -r ./docs/sphinx_requirements.txt
  $ mkdir -p ./docs/_images
  $ sphinx-build -b html ./docs ./docs/_build
