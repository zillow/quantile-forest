.. _user-guide-proximities:

Proximity Counts
----------------

Proximity counts are counts of the number of times that two samples share a leaf node. When a test set is present, the proximity counts of each sample in the test set with each sample in the training set can be computed::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> proximities = qrf.proximity_counts(X_test)  # proximity counts for test data

For each test sample, the method outputs a list of tuples of the training index and proximity count, listed in descending order by proximity count. For example, a test sample with an output of [(1, 5), (0, 3), (3, 1)], means that the test sample shared 5, 3, and 1 leaf nodes with the training samples that were (zero-)indexed as 1, 0, and 3 during model fitting, respectively.

The maximum number of proximity counts output per test sample can be limited by specifying `max_proximities`::

    >>> proximities = qrf.proximity_counts(X_test, max_proximities=10)
    >>> all([len(prox) <= 10 for prox in proximities])
    True

Out-of-bag (OOB) proximity counts can be returned by specifying `oob_score=True`::

    >>> proximities = qrf.proximity_counts(X_train, oob_score=True)
