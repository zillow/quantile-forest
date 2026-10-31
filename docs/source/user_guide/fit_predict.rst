.. _user-guide-fit-predict:

Fitting and Predicting
----------------------

Quantile forests can be fit and used to predict like standard scikit-learn estimators. In this package, the quantile forests extend standard scikit-learn forest regressors and inherent their model parameters, in addition to offering additional parameters related to quantile regression. We'll discuss many of the important model parameters below.

Fitting a Model
~~~~~~~~~~~~~~~

Let's fit a quantile forest on a simple regression dataset::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> qrf = RandomForestQuantileRegressor()
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(...)

During model initialization, the parameter `max_samples_leaf` can be specified, which determines the maximum number of samples per leaf node to retain. If `max_samples_leaf` is smaller than the number of samples in a given leaf node, then a subset of values are randomly selected. By default, the model retains one randomly selected sample per leaf node (`max_samples_leaf=1`), which enables the use of optimizations at prediction time that are not available when a variable number of samples may be retained per leaf. All samples can be retained by specifying `max_samples_leaf=None`. Note that the number of retained samples can materially impact the size of the model object.

Sample Weights
~~~~~~~~~~~~~~

Sample weights can be specified when fitting the model, as for a scikit-learn forest, which uses them to fit the trees::

    >>> import numpy as np
    >>> sample_weight = np.random.default_rng(0).uniform(0.5, 2, size=len(y_train))
    >>> qrf_weighted = RandomForestQuantileRegressor()
    >>> qrf_weighted.fit(X_train, y_train, sample_weight=sample_weight)
    RandomForestQuantileRegressor(...)

The training samples used for prediction are weighted too: a sample with weight 2 counts like two samples with weight 1, and samples with zero weight are left out. With bootstrapping (`bootstrap=True`), scikit-learn 1.9 and later draw the bootstrap samples in proportion to the weights, so each sample counts once for each time it is drawn. Otherwise, the samples in each leaf node are weighted, except that if each leaf node keeps one sample (`max_samples_leaf=1`), the sample is picked with probability proportional to its weight. With `weighted_leaves=True`, the size of a leaf node is the total weight of its samples. Either way, as without weights, the mean predictions of a model fit with `max_samples_leaf=None` equal those of a scikit-learn forest if `aggregate_leaves_first=False` (see below), and so do the medians of a model fit with `criterion="absolute_error"` (with the default `interpolation="averaged_inverted_cdf"`). Proximity counts do not include the weights.

Only the relative weights matter. The quantiles of `interpolation="inverted_cdf"` and `interpolation="averaged_inverted_cdf"` depend on nothing else, so integer weights give the same quantiles as repeating each sample that number of times. The other interpolation methods also depend on the number of samples, for which a sample of average (positive) weight counts as one sample. For an example that uses sample weights to correct for oversampling, see :ref:`gallery_plot_qrf_sample_weights`.

Making Predictions
~~~~~~~~~~~~~~~~~~

A notable advantage of quantile forests is that they can be fit once, while arbitrary quantiles can be estimated at prediction time. Accordingly, since the quantiles can be specified at prediction time, the model accepts an optional parameter during the call to the `predict` method, which can be a float or list of floats that specify the empirical quantiles to return::

    >>> y_pred = qrf.predict(X_test, quantiles=[0.25, 0.5, 0.75])
    >>> y_pred.shape[1]
    3

If the `predict` method is called without quantiles, the prediction defaults to the empirical median (`quantiles=0.5`)::

    >>> y_pred = qrf.predict(X_test)  # returns empirical median prediction

If the `predict` method is explicitly called with `quantiles="mean"`, the prediction returns the empirical mean::

    >>> y_pred = qrf.predict(X_test, quantiles="mean")  # returns mean prediction

Default quantiles can be specified at model initialization using the `default_quantiles` parameter:

    >>> qrf = RandomForestQuantileRegressor(default_quantiles=[0.25, 0.5, 0.75])
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(default_quantiles=[0.25, 0.5, 0.75])
    >>> y_pred = qrf.predict(X_test)  # predicts using the default quantiles
    >>> y_pred.ndim == 2
    True
    >>> y_pred.shape[1] == 3
    True

The default quantiles can be overwritten at prediction time by specifying a value for `quantiles`:

    >>> qrf = RandomForestQuantileRegressor(default_quantiles=[0.25, 0.5, 0.75])
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(default_quantiles=[0.25, 0.5, 0.75])
    >>> y_pred = qrf.predict(X_test, quantiles=0.5)  # uses override quantiles
    >>> y_pred.ndim == 1
    True

The output of the `predict` method is an array with one column for each specified quantile or a single column if an individual quantile is specified. The order of the output columns corresponds to the order of the quantiles, which can be specified in any order (i.e., they do not need to be monotonically ordered)::

    >>> y_pred = qrf.predict(X_test, quantiles=[0.5, 0.25, 0.75])
    >>> bool((y_pred[:, 0] >= y_pred[:, 1]).all())
    True

By default (`interpolation="averaged_inverted_cdf"`), a quantile `q` is the smallest training target such that at least a fraction `q` of the (weighted) training samples are at or below it, or, if exactly that fraction are, its average with the next target, so that the median is the usual median. This is NumPy's method of the same name and how scikit-learn calculates weighted quantiles. Other methods can be specified with `interpolation`, such as linear interpolation between training targets, the default before version 1.5::

    >>> y_pred = qrf.predict(X_test, interpolation="linear")

For a comparison of the methods, see :ref:`gallery_plot_qrf_interpolation_methods`.

Multi-target quantile regression is also supported. If the target values are multi-dimensional, then the second output column will correspond to the number of targets::

    >>> from sklearn import datasets
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.make_regression(n_samples=100, n_targets=2, random_state=0)
    >>> y.shape
    (100, 2)
    >>> qrf_multi = RandomForestQuantileRegressor()
    >>> qrf_multi.fit(X, y)
    RandomForestQuantileRegressor()
    >>> quantiles = [0.25, 0.5, 0.75]
    >>> y_pred = qrf_multi.predict(X, quantiles=quantiles)
    >>> y_pred.ndim == 3
    True
    >>> y_pred.shape[0] == len(X)
    True
    >>> y_pred.shape[2] == len(quantiles)
    True
    >>> y_pred.shape[1] == y.shape[1]  # number of targets
    True

Quantile Weighting
~~~~~~~~~~~~~~~~~~

By default, the predict method calculates quantiles using a weighted quantile method (`weighted_quantile=True`), which assigns a weight to each sample in the training set based on the number of times that it co-occurs in the same leaves as the test sample. When the number of samples in the training set is larger than the expected number of co-occurring samples across all trees, it can be more efficient to calculate an unweighted quantile (`weighted_quantile=False`), which aggregates a list of training `y` values for each leaf node to which the test sample belongs across all trees. For a given input, both methods can return the same output values::

    >>> import numpy as np
    >>> y_pred_weighted = qrf.predict(X_test, weighted_quantile=True)
    >>> y_pred_unweighted = qrf.predict(X_test, weighted_quantile=False)
    >>> np.allclose(y_pred_weighted, y_pred_unweighted)
    True

By default, the predict method calculates quantiles by giving each sample in a leaf (including repeated bootstrap samples) equal weight (`weighted_leaves=False`). If `weighted_leaves=True`, each sample will be weighted inversely according to the size of its leaf node. Note that this leaf-based weighting can only be used with weighted quantiles.

Out-of-Bag Estimation
~~~~~~~~~~~~~~~~~~~~~

Out-of-bag (OOB) predictions can be returned by specifying `oob_score=True`::

    >>> y_pred_oob = qrf.predict(X_train, quantiles=0.5, oob_score=True)

By default, when the `predict` method is called with the OOB flag set to True, it assumes that the input samples are the training samples, arranged in the same order as during model fitting. It accepts an optional parameter that can be used to specify the training index of each input sample, with -1 used to specify non-training samples that can in effect be scored in-bag (IB) during the same call::

    >>> import numpy as np
    >>> X_mixed = np.concatenate([X_train, X_test])
    >>> indices = np.concatenate([np.arange(len(X_train)), np.full(len(X_test), -1)])
    >>> kwargs = {"oob_score": True, "indices": indices}
    >>> y_pred_mix = qrf.predict(X_mixed, quantiles=[0.25, 0.5, 0.75], **kwargs)
    >>> y_pred_train_oob = y_pred_mix[:len(X_train)]  # training predictions are OOB
    >>> y_pred_test = y_pred_mix[-len(X_test):]  # new test data predictions are IB

This allows all samples, both from the training and test sets, to be scored with a single call to `predict`, whereby OOB predictions are returned for the training samples and IB (i.e., non-OOB) predictions are returned for the test samples.

Random Forest Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

The predictions of a standard random forest can also be recovered from a quantile forest without retraining when initialized with `max_samples_leaf=None`. This can be accomplished at inference time by passing `quantiles="mean"` (or `quantiles=0.5` if the model was specifically fitted with `criterion="absolute_error"`) and `aggregate_leaves_first=False`, the latter which specifies a Boolean flag to average the leaf values before aggregating the leaves across trees. This configuration essentially replicates the prediction process used by a standard random forest regressor, which is an averaging of mean (or median) leaf values across trees::

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> rf = RandomForestRegressor(random_state=0)
    >>> qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0)
    >>> rf.fit(X_train, y_train)
    RandomForestRegressor(random_state=0)
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0)
    >>> kwargs = {"quantiles": "mean", "aggregate_leaves_first": False}
    >>> y_pred_rf = rf.predict(X_test)
    >>> y_pred_qrf = qrf.predict(X_test, **kwargs)
    >>> np.allclose(y_pred_rf, y_pred_qrf)
    True

User-Specified Functions
~~~~~~~~~~~~~~~~~~~~~~~~

While a QRF is designed to estimate quantiles from the empirical distribution calculated for each sample, in many cases it may be useful to use the empirical distribution to calculate other quantities of interest. For more details, see :ref:`gallery_plot_qrf_custom_functions`.

Monotonic Constraints
~~~~~~~~~~~~~~~~~~~~~

Predictions can be constrained to increase (1) or decrease (-1) monotonically with individual features by specifying `monotonic_cst`, which requires `max_samples_leaf=1` (the default)::

    >>> monotonic_cst = [1] + [0] * (X_train.shape[1] - 1)  # increasing in the first feature
    >>> qrf = RandomForestQuantileRegressor(monotonic_cst=monotonic_cst)
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(...)

Each tree is kept monotonic by clipping the target value that each leaf keeps: a split on a constrained feature bounds the leaves on either side by the midpoint of their mean targets. Because the bounds are derived from means, they also clip much of the variation across trees that quantile predictions rely on, so the predicted distributions are much narrower than without constraints. For example, with 2,000 training samples whose targets equal a feature plus standard normal noise, `min_samples_leaf=5`, and an increasing constraint on that feature, 80% prediction intervals covered 31% of new targets, against 74% without the constraint. Monotonic constraints are therefore best suited to point predictions, such as the median or mean. If prediction intervals are needed, calibrate them on held-out data (e.g., with conformal prediction), or fit without the constraint.
