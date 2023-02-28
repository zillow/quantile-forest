# quantile-forest

[![PyPI - Version](https://img.shields.io/pypi/v/quantile-forest)](https://pypi.org/project/quantile-forest)
[![License](https://img.shields.io/github/license/zillow/quantile-forest)](https://github.com/zillow/quantile-forest/blob/main/LICENSE)
[![GitHub Actions](https://github.com/zillow/quantile-forest/actions/workflows/build.yml/badge.svg)](https://github.com/zillow/quantile-forest/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/zillow/quantile-forest/branch/main/graph/badge.svg?token=STRT8T67YP)](https://codecov.io/gh/zillow/quantile-forest)
[![Code Style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/quantile-forest)](https://pypi.org/project/quantile-forest)

**quantile-forest** offers a Python implementation of quantile regression forests compatible with scikit-learn.

Quantile regression forests are a non-parametric, tree-based ensemble method for estimating conditional quantiles, with application to high-dimensional data and uncertainty estimation [[1]](#1). The estimators in this package extend the forest estimators available in scikit-learn to estimate conditional quantiles. They are compatible with and can serve as drop-in replacements for the scikit-learn variants.

#### Example of fitted model predictions and prediction intervals on California housing data ([code](https://zillow.github.io/quantile-forest/auto_examples/plot_quantile_regression_intervals.html#sphx-glr-auto-examples-plot-quantile-regression-intervals-py))
<img src="https://zillow.github.io/quantile-forest/_images/sphx_glr_plot_quantile_regression_intervals_001.png" height="300" />

Quick Start
-----------

Install quantile-forest from [PyPI](https://pypi.org/project/quantile-forest) using `pip`:

```bash
pip install quantile-forest
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
