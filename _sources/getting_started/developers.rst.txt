.. _developers:

Developer's Guide
-----------------

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Building the package from source additionally requires the following dependencies:

* cython (>=3.0a4)

We also use pre-commit hooks to identify simple issues before submission.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development installation, we recommend using `uv <https://github.com/astral-sh/uv>`_:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/zillow/quantile-forest.git
      cd quantile-forest

2. Create a virtual environment:

   .. code-block:: bash

      uv venv

3. Build the package:

   .. code-block:: bash

      uv pip install --editable . --verbose

4. Install and configure pre-commit hooks:

   .. code-block:: bash

      uv pip install pre-commit
      pre-commit install

   You can run the hooks manually on all files with:

   .. code-block:: bash

      pre-commit run --all-files

5. Run the test suite to verify the installation:

   .. code-block:: bash

      uv run pytest

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

  uv pip install pytest pytest-cov

To test the code::

  uv run pytest quantile_forest -v

To test the code and produce a coverage report::

  uv run pytest quantile_forest --cov-report html --cov=quantile_forest

To test the documentation::

  uv run pytest --doctest-glob="*.rst" --doctest-modules docs

Documentation
~~~~~~~~~~~~~

To build the documentation, run::

  uv pip install -r ./docs/sphinx_requirements.txt
  mkdir -p ./docs/source/_images
  uv run sphinx-build -b html ./docs/source ./docs/_build
