# quantile-forest <a href="https://zillow.github.io/quantile-forest/"><img align="right" src="https://zillow.github.io/quantile-forest/_static/quantile-forest-logo.svg" height="50"></img></a>

[![PyPI - Version](https://img.shields.io/pypi/v/quantile-forest)](https://pypi.org/project/quantile-forest)
[![License](https://img.shields.io/github/license/zillow/quantile-forest)](https://github.com/zillow/quantile-forest/blob/main/LICENSE)
[![GitHub Actions](https://github.com/zillow/quantile-forest/actions/workflows/build.yml/badge.svg)](https://github.com/zillow/quantile-forest/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/zillow/quantile-forest/branch/main/graph/badge.svg?token=STRT8T67YP)](https://codecov.io/gh/zillow/quantile-forest)
[![Code Style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05976/status.svg)](https://doi.org/10.21105/joss.05976)

**quantile-forest** offers a Python implementation of quantile regression forests compatible with scikit-learn.

Quantile regression forests (QRF) are a non-parametric, tree-based ensemble method for estimating conditional quantiles, with application to high-dimensional data and uncertainty estimation [[1]](#1). The estimators in this package are performant, Cython-optimized QRF implementations that extend the forest estimators available in scikit-learn to estimate conditional quantiles. The estimators can estimate arbitrary quantiles at prediction time without retraining and provide methods for out-of-bag estimation, calculating quantile ranks, and computing proximity counts. They are compatible with and can serve as drop-in replacements for the scikit-learn forest regressors.

#### Example of fitted model predictions and prediction intervals on California housing data ([code](https://zillow.github.io/quantile-forest/gallery/plot_qrf_prediction_intervals.html))
<img src="https://zillow.github.io/quantile-forest/_static/plot_qrf_prediction_intervals.png"/>

Quick Start
-----------

To install quantile-forest from [PyPI](https://pypi.org/project/quantile-forest) using `pip`:

```bash
pip install quantile-forest
```

To install quantile-forest from [conda-forge](https://anaconda.org/conda-forge/quantile-forest) using `conda`:

```bash
conda install quantile-forest -c conda-forge
```

Usage
-----

```python
from quantile_forest import RandomForestQuantileRegressor
from sklearn import datasets
X, y = datasets.fetch_california_housing(return_X_y=True)
qrf = RandomForestQuantileRegressor()
qrf.fit(X, y)
y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975])
```

Documentation
-------------

An installation guide, API documentation, and examples can be found in the [documentation](https://zillow.github.io/quantile-forest).


References
----------

<a id="1">[1]</a> N. Meinshausen, "Quantile Regression Forests", Journal of Machine Learning Research, 7(Jun), 983-999, 2006. http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

Citation
--------

If you use this package in academic work, please consider citing https://joss.theoj.org/papers/10.21105/joss.05976:

```bib
@article{Johnson2024,
    doi = {10.21105/joss.05976},
    url = {https://doi.org/10.21105/joss.05976},
    year = {2024},
    publisher = {The Open Journal},
    volume = {9},
    number = {93},
    pages = {5976},
    author = {Reid A. Johnson},
    title = {quantile-forest: A Python Package for Quantile Regression Forests},
    journal = {Journal of Open Source Software}
}
```
