.. title:: User Guide

.. _user_guide:

==========
User Guide
==========

Introduction
------------

Random forests have proven to be very popular and powerful for regression and classification. For regression, random forests give an accurate approximation of the conditional mean of a response variable. That is, if we let :math:`Y` be a real-valued response variable and :math:`X` a covariate or predictor variable, they estimate :math:`E(Y | X)`, which can be interpreted as the expected value of the output :math:`Y` given the input :math:`X`.

However random forests provide information about the full conditional distribution of the response variable, not only about the conditional mean. Quantile regression forests, a generalization of random forests, can be used to infer conditional quantiles. That is, they return :math:`y` at :math:`q` for which :math:`F(Y=y|X) = q`, where :math:`q` is the quantile.

The quantiles give more complete information about the distribution of :math:`Y` as a function of the predictor variable :math:`X` than the conditional mean alone. They can be useful, for example, to build prediction intervals or to perform outlier detection in a high-dimensional dataset.

In practice, the empirical estimation of quantiles can be calculated in several ways. In this package, a desired quantile is calculated from the input rank :math:`x` such that :math:`x = (N + 1 - 2C)q + C`, where :math:`q` is the quantile, :math:`N` is the number of samples, and :math:`C` is a constant (degree of freedom). In this package, :math:`C = 1`. This package provides methods that calculate quantiles using samples that are weighted and unweighted. In a weighted quantile, :math:`N` is calculated from the fraction of the total weight instead of the total number of samples.

Quantile Regression Forests
---------------------------

A standard decision tree can be extended in a straightforward way to estimate conditional quantiles. When a decision tree is fit, rather than storing only the sufficient statistics of the response variable at the leaf node, such as the mean and variance, all of the response values can be stored with the leaf node. At prediction time, these values can then be used to calculate empirical quantile estimates.

The quantile-based approach can be extended to random forests. To estimate :math:`F(Y=y|x) = q`, each response value in `y_train` is given a weight or frequency. Formally, the weight or frequency given to the :math:`j`\th sample of `y_train`, :math:`y_j`, while estimating the quantile is

.. math::

  \frac{1}{T} \sum_{t=1}^{T} \frac{\mathbb{1}(y_j \in L(x))}{\sum_{i=1}^N \mathbb{1}(y_i \in L(x))},

where :math:`L(x)` denotes the leaf that :math:`x` falls into.

Informally, this means that given a new unknown sample, we first find the leaf that it falls into at each tree. Then for each `(X, y)` pair in the training data, a weight or frequency is given to `y` for each tree based on the number of times each training sample falls into the same leaf as the new sample. This information can then be used to calculate the empirical quantile estimates.

This approach was first proposed by :cite:t:`2006:meinshausen`.

Fitting and Predicting
----------------------
Quantile forests can be fit and used to predict like standard scikit-learn estimators.

Let's fit a quantile forest on a simple regression dataset::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> reg = RandomForestQuantileRegressor()
    >>> reg.fit(X_train, y_train)
    RandomForestQuantileRegressor(...)

During model initialization, the parameter `max_samples_leaf` can be specified, which determines the maximum number of samples per leaf node to retain. If `max_samples_leaf` is smaller than the number of samples in a given leaf node, then a subset of values are randomly selected. By default, the model retains one randomly selected sample per leaf node (`max_samples_leaf = 1`); all samples can be retained by specifying `max_samples_leaf = None`. Note that the number of retained samples can materially impact the size of the model object.

A notable advantage of quantile forests is that they can be fit once, while arbitrary quantiles can be estimated at prediction time. Accordingly, since the quantiles can be specified at prediction time, the model accepts an optional parameter during the call to the `predict` method, which can be a float or list of floats that specify the empirical quantiles to return::

    >>> y_pred = reg.predict(X_test, quantiles=[0.25, 0.5, 0.75])  # returns three columns per row

If the `predict` method is called without quantiles, the prediction defaults to the empirical median (`quantiles = 0.5`)::

    >>> y_pred = reg.predict(X_test)  # returns empirical median prediction for each test sample

If the `predict` method is explicitly called with `quantiles = None`, the prediction returns the empirical mean::

    >>> y_pred = reg.predict(X_test, quantiles=None)  # returns empirical mean prediction for each test sample

The output of the `predict` method is an array with one column for each specified quantile or a single column if no quantiles are specified. The order of the output columns corresponds to the order of the quantiles, which can be specified in any order (i.e., they do not need to be monotonically ordered)::

    >>> y_pred = reg.predict(X_test, quantiles=[0.5, 0.25, 0.75])  # first column corresponds to quantile 0.5

By default, the predict method calculates quantiles by weighting each sample inversely according to the size of its leaf node (`weighted_leaves = True`). If `weighted_leaves = False`, each sample in a leaf (including repeated bootstrap samples) will be given equal weight. Note that this leaf-based weighting can only be used with weighted quantiles.

By default, the predict method calculates quantiles using a weighted quantile method (`weighted_quantile = True`), which assigns a weight to each sample in the training set based on the number of times that it co-occurs in the same leaves as the test sample. When the number of samples in the training set is larger than the expected size of this list (i.e., :math:`n_{train} \gg n_{trees} \cdot n_{leaves} \cdot n_{leafsamples}`), it can be more efficient to calculate an unweighted quantile (`weighted_quantile = False`), which aggregates the list of training `y` values for each leaf node to which the test sample belongs across all trees. For a given input, both methods can return the same output values::

    >>> import numpy as np
    >>> y_pred_weighted = reg.predict(X_test, weighted_quantile=True, weighted_leaves=False)  # weighted quantile (default)
    >>> y_pred_unweighted = reg.predict(X_test, weighted_quantile=False, weighted_leaves=False)  # unweighted quantile
    >>> np.allclose(y_pred_weighted, y_pred_unweighted)
    True

Out-of-bag (OOB) predictions can be returned by specifying `oob_score = True`::

    >>> y_pred_oob = reg.predict(X_train, quantiles=[0.25, 0.5, 0.75], oob_score=True)

By default, when the `predict` method is called with the OOB flag set to True, it assumes that the input samples are the training samples, arranged in the same order as during model fitting. It accepts an optional parameter that can be used to specify the training index of each input sample, with -1 used to specify non-training samples that can in effect be scored in-bag (IB) during the same call::

    >>> import numpy as np
    >>> X_mixed = np.concatenate([X_train, X_test])
    >>> indices = np.concatenate([np.arange(len(X_train)), np.full(len(X_test), -1, dtype=int)])
    >>> y_pred_mix = reg.predict(X_mixed, quantiles=[0.25, 0.5, 0.75], oob_score=True, indices=indices)
    >>> y_pred_train_oob = y_pred_mix[:len(X_train)]  # predictions on the training data are OOB
    >>> y_pred_test = y_pred_mix[-len(X_test):]  # predictions on the new test data are IB

This allows all samples, both from the training and test sets, to be scored with a single call to `predict`, whereby OOB predictions are returned for the training samples and IB (i.e., non-OOB) predictions are returned for the test samples.

The predictions of a standard random forest can also be recovered from a quantile forest without retraining by passing `quantiles = None` and `aggregate_leaves_first = False`, the latter which specifies a Boolean flag to average the leaf values before aggregating the leaves across trees. This configuration essentially replicates the prediction process used by a standard random forest regressor, which is an averaging of mean leaf values across trees::

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> rf = RandomForestRegressor(random_state=0)
    >>> qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0)
    >>> rf.fit(X_train, y_train), qrf.fit(X_train, y_train)
    (RandomForestRegressor(random_state=0), RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0))
    >>> y_pred_rf = rf.predict(X_test)
    >>> y_pred_qrf = qrf.predict(X_test, quantiles=None, aggregate_leaves_first=False)
    >>> np.allclose(y_pred_rf, y_pred_qrf)
    True

Quantile Ranks
----------------

The quantile rank of a score is the quantile of scores in its frequency distribution that are equal to or lower than it. The output quantile rank will be a value in the range [0, 1] for each test sample. The quantile rank of each sample is calculated by aggregating all of the training samples that share the same leaf node across all of the trees::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> reg = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> y_ranks = reg.quantile_ranks(X_test, y_test)  # quantile ranks of y_test

Out-of-bag (OOB) quantile ranks can be returned by specifying `oob_score = True`::

    >>> y_ranks_oob = reg.quantile_ranks(X_train, y_train, oob_score=True)

Proximity Counts
----------------
Proximity counts are counts of the number of times that two samples share a leaf node. When a test set is present, the proximity counts of each sample in the test set with each sample in the training set can be computed::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> reg = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> proximities = reg.proximity_counts(X_test)

For each test sample, the method outputs a list of tuples of the training index and proximity count, listed in descending order by proximity count. For example, a test sample with an output of [(1, 5), (0, 3), (3, 1)], means that the test sample shared 5, 3, and 1 leaf nodes with the training samples that were (zero-)indexed as 1, 0, and 3 during model fitting, respectively.

The maximum number of proximity counts output per test sample can be limited by specifying `max_proximities`::

    >>> proximities = reg.proximity_counts(X_test, max_proximities=10)

Out-of-bag (OOB) proximity counts can be returned by specifying `oob_score = True`::

    >>> proximities = reg.proximity_counts(X_train, oob_score=True)

References
----------
.. bibliography::
