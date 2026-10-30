.. _user-guide-proximities:

Proximity Counts
----------------

Proximity counts measure how often a quantile forest puts a sample in the same leaf nodes as each training sample. When a test set is present, the proximity counts of each sample in the test set with each sample in the training set can be computed::

    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from quantile_forest import RandomForestQuantileRegressor
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> qrf = RandomForestQuantileRegressor().fit(X_train, y_train)
    >>> proximities = qrf.proximity_counts(X_test)  # proximity counts for test data

For each test sample, the method outputs a list of tuples of the training index and proximity count, listed in descending order by proximity count. The proximity count of a training sample is the number of times that it is among the samples kept in the leaf nodes that the test sample falls into, summed across the trees. For example, a test sample with an output of [(1, 5), (0, 3), (3, 1)] means that the training samples that were (zero-)indexed as 1, 0, and 3 during model fitting were kept 5, 3, and 1 times, respectively, in the leaf nodes that the test sample falls into.

By default (`max_samples_leaf=1`), each leaf node keeps one randomly selected training sample, so each tree adds one count, and the counts of a test sample add up to the number of trees::

    >>> sum(count for _, count in proximities[0])
    100

So the counts cover at most as many training samples as there are trees, and a training sample that shares a leaf node with the test sample counts only if the leaf node kept it. To count all of the training samples in the leaf nodes, keep them all by setting `max_samples_leaf=None`::

    >>> qrf = RandomForestQuantileRegressor(max_samples_leaf=None)
    >>> qrf.fit(X_train, y_train)
    RandomForestQuantileRegressor(max_samples_leaf=None)
    >>> proximities = qrf.proximity_counts(X_test)

The counts also depend on which training samples each tree uses:

- With bootstrapping (`bootstrap=True`), each tree keeps only the samples drawn for it, and a sample drawn more than once is counted once for each draw.
- Samples with zero or negative weight are not kept.

For a model fit without sample weights, the proximity counts are the weights that the `predict` method gives the training samples by default. They differ from the `proximities described by Breiman <https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox>`_, which put all of the training samples down each tree, whether or not they were drawn for it, add one for each tree in which two samples share a leaf node, and divide by the number of trees. With `bootstrap=False` and `max_samples_leaf=None`, each proximity count is the number of trees in which the two samples share a leaf node. Breiman's proximities can also be calculated from the leaf nodes returned by the `apply` method::

    >>> X_train_leaves = qrf.apply(X_train)
    >>> X_test_leaves = qrf.apply(X_test)
    >>> same_leaf = X_test_leaves[:, None, :] == X_train_leaves[None, :, :]
    >>> proximity_matrix = same_leaf.mean(axis=2)
    >>> proximity_matrix.shape == (len(X_test), len(X_train))
    True

The maximum number of proximity counts output per test sample can be limited by specifying `max_proximities`::

    >>> proximities = qrf.proximity_counts(X_test, max_proximities=10)
    >>> all([len(prox) <= 10 for prox in proximities])
    True

Out-of-bag (OOB) proximity counts can be returned by specifying `oob_score=True`::

    >>> proximities = qrf.proximity_counts(X_train, oob_score=True)
