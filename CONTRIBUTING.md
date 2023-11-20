# Contributions

Contributions are welcome, encouraged, and appreciated!

If you encounter any bugs while using `quantile-forest`, or believe there's a feature that would prove useful, feel free to [submit a new issue](https://github.com/zillow/quantile-forest/issues/new/choose).

All contributions, suggestions, and feedback you submitted are accepted under the [project's license](https://github.com/zillow/quantile-forest/blob/main/LICENSE).

## Setting Up Your Environment

To contribute to the `quantile-forest` source code, start by forking and then cloning the repository (i.e. `git clone git@github.com:YourUsername/quantile-forest.git`)

Once inside the repository, to build and install the package, run:

```cmd
python setup.py build_ext --inplace
python setup.py install
 ```

## Testing Your Changes

To execute unit tests from the `quantile-forest` repository, run:

```cmd
pytest quantile_forest -v
```

## Troubleshooting

If the build fails because SciPy is not installed, ensure OpenBLAS and LAPACK are available and accessible.

On macOS, run:

```cmd
brew install openblas
brew install lapack
export SYSTEM_VERSION_COMPAT=1
```

Then try rebuilding.
