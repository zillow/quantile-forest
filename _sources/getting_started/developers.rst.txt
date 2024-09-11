.. _developers:

Developer's Guide
-----------------

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~
Building the package from source additionally requires the following dependencies:

* cython (>=3.0a4)

We also use pre-commit hooks to identify simple issues before submission.

To install these dependencies, run::

  pip install cython pre-commit

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

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

Ensure that `pytest` and `pytest-cov` are installed::

  $ pip install pytest pytest-cov

To test the code::

  $ python -m pytest quantile_forest -v

To test the code and produce a coverage report::

  $ python -m pytest quantile_forest --cov-report html --cov=quantile_forest

To test the documentation::

  $ python -m pytest --doctest-glob="*.rst" --doctest-modules docs

Documentation
~~~~~~~~~~~~~

To build the documentation, run::

  $ pip install -r ./docs/sphinx_requirements.txt
  $ mkdir -p ./docs/_images
  $ sphinx-build -b html ./docs ./docs/_build
