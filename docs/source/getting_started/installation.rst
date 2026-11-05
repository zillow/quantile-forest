.. _install:

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

The quantile-forest package requires the following dependencies:

* python (>=3.9)
* numpy (>=1.23)
* scikit-learn (>=1.5)
* scipy (>=1.4)

Installation
~~~~~~~~~~~~

quantile-forest can be installed using `pip` or `conda`.

Using `pip`::

  pip install quantile-forest

Using `conda`::

  conda install quantile-forest -c conda-forge

Usage
~~~~~

A quantile forest is fit like a scikit-learn regressor, and predicts any quantiles of the target without being fit again. For example, to predict the median and a 95% prediction interval::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> y_pred = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])
    >>> y_pred.shape
    (111, 3)

The :ref:`user guide <user-guide-intro>` covers fitting and predicting in more detail, and the :ref:`examples <example-gallery>` show quantile forests in use.

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :hidden:

   self
   developers
