.. _user-guide-quantile-ranks:

Quantile Ranks
--------------

The quantile rank is the fraction of scores in a frequency distribution that are less than (or equal to) that score. For a quantile forest, the frequency distribution is the set of training sample response values that are used to construct the empirical quantile estimates. The quantile rank of each sample is calculated by aggregating the response values from all of the training samples that share the same leaf node across all of the trees. The output quantile rank will be a value in the range [0, 1] for each test sample::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> y_ranks = qrf.quantile_ranks(X_test, y_test)  # quantile ranks for test data

Out-of-bag (OOB) quantile ranks can be returned by specifying `oob_score=True`::

    >>> y_ranks_oob = qrf.quantile_ranks(X_train, y_train, oob_score=True)
