"""
Testing for the quantile forest module (quantile_forest._quantile_forest).
"""

import math
import warnings
from typing import Any, Dict

import numpy as np
import pytest
from scipy.stats import percentileofscore
from sklearn import datasets
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)
from sklearn.utils.validation import check_random_state

from quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor
from quantile_forest._quantile_forest_fast import (
    calc_mean,
    calc_quantile,
    calc_quantile_rank,
    calc_weighted_mean,
    calc_weighted_quantile,
    generate_unsampled_indices,
)

np.random.seed(0)

rng = check_random_state(0)

# Load the California Housing Prices dataset.
california = datasets.fetch_california_housing()
perm = rng.permutation(min(california.target.size, 500))
X_california = california.data[perm]
y_california = california.target[perm]

FOREST_REGRESSORS: Dict[str, Any] = {
    "ExtraTreesQuantileRegressor": ExtraTreesQuantileRegressor,
    "RandomForestQuantileRegressor": RandomForestQuantileRegressor,
}


def check_regression_toy(name, weighted_quantile):
    """Check quantile regression on a toy dataset."""
    quantiles = [0.25, 0.5, 0.75]

    # Toy sample.
    X = [[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]]
    y = [-1, -1, 0, 1, 1, 2]
    y_test = [[-1, -1], [2, 2], [3, 2]]

    ForestRegressor = FOREST_REGRESSORS[name]

    regr = ForestRegressor(n_estimators=10, max_samples_leaf=None, bootstrap=False, random_state=0)
    regr.fit(X, y)

    # Check model and apply outputs shape.
    leaf_indices = regr.apply(X)
    assert leaf_indices.shape == (len(X), regr.n_estimators)
    assert 10 == len(regr)

    # Check aggregated quantile predictions.
    y_true = [[0.0, 0.5, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = regr.predict(
            y_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=True,
        )
    assert_allclose(y_pred, y_true)

    # Check unaggregated quantile predictions.
    y_true = [[0.25, 0.5, 0.75], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = regr.predict(
            y_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=False,
        )
    assert_allclose(y_pred, y_true)

    assert regr._more_tags()


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("weighted_quantile", [True, False])
def test_regression_toy(name, weighted_quantile):
    check_regression_toy(name, weighted_quantile)


def check_california_criterion(name, criterion):
    # Check for consistency on the California Housing Prices dataset.
    ForestRegressor = FOREST_REGRESSORS[name]

    regr = ForestRegressor(n_estimators=5, criterion=criterion, max_features=None, random_state=0)
    regr.fit(X_california, y_california)
    score = regr.score(X_california, y_california, quantiles=0.5)
    assert score > 0.9, f"Failed with max_features=None, criterion {criterion} and score={score}."

    # Test maximum features.
    regr = ForestRegressor(n_estimators=5, criterion=criterion, max_features=6, random_state=0)
    regr.fit(X_california, y_california)
    score = regr.score(X_california, y_california, quantiles=0.5)
    assert score > 0.9, f"Failed with max_features=6, criterion {criterion} and score={score}."

    # Test sample weights.
    regr = ForestRegressor(n_estimators=5, criterion=criterion, random_state=0)
    sample_weight = np.ones(y_california.shape)
    regr.fit(X_california, y_california, sample_weight=sample_weight)
    score = regr.score(X_california, y_california, quantiles=0.5)
    assert score > 0.9, f"Failed with criterion {criterion}, sample weight and score={score}."


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("criterion", ("squared_error", "absolute_error", "friedman_mse"))
def test_california(name, criterion):
    check_california_criterion(name, criterion)


def check_predict_quantiles_toy(name):
    # Check quantile predictions on toy data.
    quantiles = [0.25, 0.5, 0.75]

    ForestRegressor = FOREST_REGRESSORS[name]

    # Check predicted quantiles on toy sample.
    X = [[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]]
    y = [-1, -1, 0, 1, 1, 2]

    est = ForestRegressor(n_estimators=1, max_samples_leaf=None, bootstrap=False, random_state=0)
    est.fit(X, y)

    expected = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    y_pred = est.predict(X, quantiles, interpolation="lower")
    assert_array_equal(y_pred, expected)

    expected = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    y_pred = est.predict(X, quantiles, interpolation="higher")
    assert_array_equal(y_pred, expected)

    expected = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    y_pred = est.predict(X, quantiles, interpolation="midpoint")
    assert_array_equal(y_pred, expected)

    expected = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    y_pred = est.predict(X, quantiles, interpolation="nearest")
    assert_array_equal(y_pred, expected)

    expected = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.25, 0.5, 0.75],
        [0.25, 0.5, 0.75],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    y_pred = est.predict(X, quantiles, interpolation="linear")
    assert_array_equal(y_pred, expected)

    for oob_score in [False, True]:
        est = ForestRegressor(
            n_estimators=25,
            max_depth=1,
            max_samples_leaf=None,
            bootstrap=True,
            random_state=0,
        )
        est.fit(X, y)

        # Check that weighted and unweighted quantiles are approximately equal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_pred1 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=True,
                weighted_leaves=False,
                oob_score=oob_score,
            )
            y_pred2 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=False,
                weighted_leaves=False,
                oob_score=oob_score,
            )
        assert_allclose(y_pred1, y_pred2)

        # Check that weighted and unweighted leaves are not equal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_pred1 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=True,
                weighted_leaves=True,
                oob_score=oob_score,
            )
            y_pred2 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=True,
                weighted_leaves=False,
                oob_score=oob_score,
            )
        assert_raises(AssertionError, assert_allclose, y_pred1, y_pred2)

        # Check that leaf weighting without weighted quantiles does nothing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_pred1 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=False,
                weighted_leaves=True,
                oob_score=oob_score,
            )
            y_pred2 = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=False,
                weighted_leaves=False,
                oob_score=oob_score,
            )
        assert_array_equal(y_pred1, y_pred2)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_predict_quantiles_toy(name):
    check_predict_quantiles_toy(name)


def check_predict_quantiles(
    name,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    ForestRegressor = FOREST_REGRESSORS[name]

    # Check predicted quantiles on (semi-)random data.
    x1 = np.random.choice(np.arange(0, 101), size=1000)
    e1_high = 10
    e1 = np.random.uniform(0, e1_high)

    X_train = x1.reshape(-1, 1)
    y_train = np.squeeze(x1 * 2 + e1)

    x2 = np.random.choice(np.arange(0, 101), size=4)

    X_test = x2.reshape(-1, 1)
    y_test = np.squeeze(X_test * 2)

    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    est = ForestRegressor(n_estimators=10, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = est.predict(
            X_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
        )
    if isinstance(quantiles, list):
        assert y_pred.shape == (X_test.shape[0], len(quantiles))
        assert_array_almost_equal(y_pred[:, 1], y_test, -e1_high)
    else:
        assert y_pred.shape == (X_test.shape[0],)
        assert_array_almost_equal(y_pred, y_test, -e1_high)

    # Check predicted quantiles on the California Housing Prices dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X_california, y_california, test_size=0.25, random_state=0
    )

    est = ForestRegressor(n_estimators=10, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = est.predict(
            X_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
        )
    if isinstance(quantiles, list):
        assert y_pred.shape == (X_test.shape[0], len(quantiles))
    else:
        assert y_pred.shape == (X_test.shape[0],)

    if isinstance(quantiles, list):
        # Check that predicted values are monotonic.
        assert np.all(np.less_equal(y_pred[:, 0], y_pred[:, 1]))
        assert np.all(np.less_equal(y_pred[:, 1], y_pred[:, 2]))

    # Check that weighted and unweighted quantiles are all equal.
    est = ForestRegressor(n_estimators=10, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    y_pred_1 = est.predict(
        X_test,
        quantiles=quantiles,
        weighted_quantile=True,
        weighted_leaves=False,
        aggregate_leaves_first=aggregate_leaves_first,
    )
    y_pred_2 = est.predict(
        X_test,
        quantiles=quantiles,
        weighted_quantile=False,
        weighted_leaves=False,
        aggregate_leaves_first=aggregate_leaves_first,
    )
    assert_allclose(y_pred_1, y_pred_2)

    # Check that weighted and unweighted leaves are all equal.
    est = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_1 = est.predict(
            X_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            weighted_leaves=True,
            aggregate_leaves_first=False,
        )
        y_pred_2 = est.predict(
            X_test,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            weighted_leaves=False,
            aggregate_leaves_first=False,
        )
    assert_allclose(y_pred_1, y_pred_2)

    # Check that aggregated and unaggregated quantiles are all equal.
    est = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    y_pred_1 = est.predict(
        X_test,
        quantiles=quantiles,
        weighted_quantile=weighted_quantile,
        weighted_leaves=False,
        aggregate_leaves_first=True,
    )
    y_pred_2 = est.predict(
        X_test,
        quantiles=quantiles,
        weighted_quantile=weighted_quantile,
        weighted_leaves=False,
        aggregate_leaves_first=False,
    )
    assert_allclose(y_pred_1, y_pred_2)

    # Check that omitting quantiles is the same as setting to 0.5.
    est = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    y_pred_1 = est.predict(
        X_test,
        weighted_quantile=weighted_quantile,
        weighted_leaves=False,
        aggregate_leaves_first=True,
    )
    y_pred_2 = est.predict(
        X_test,
        quantiles=0.5,
        weighted_quantile=weighted_quantile,
        weighted_leaves=False,
        aggregate_leaves_first=False,
    )
    assert_allclose(y_pred_1, y_pred_2)

    # Check that unaggregated predicted means match non-quantile regressor.
    if not aggregate_leaves_first:
        if name == "ExtraTreesQuantileRegressor":
            est1 = ExtraTreesRegressor(
                n_estimators=1 if max_samples_leaf == 1 else 10,
                random_state=0,
            )
            est2 = ExtraTreesQuantileRegressor(
                n_estimators=1 if max_samples_leaf == 1 else 10,
                default_quantiles=None,
                max_samples_leaf=max_samples_leaf,
                random_state=0,
            )
        else:
            est1 = RandomForestRegressor(
                n_estimators=1 if max_samples_leaf == 1 else 10,
                random_state=0,
            )
            est2 = RandomForestQuantileRegressor(
                n_estimators=1 if max_samples_leaf == 1 else 10,
                default_quantiles=None,
                max_samples_leaf=max_samples_leaf,
                random_state=0,
            )
        y_pred_1 = est1.fit(X_train, y_train).predict(X_test)
        y_pred_2 = est2.fit(X_train, y_train).predict(
            X_test,
            quantiles=None,
            weighted_quantile=weighted_quantile,
            weighted_leaves=False,
            aggregate_leaves_first=False,
        )
        assert_allclose(y_pred_1, y_pred_2)

        # Check multi-target outputs.
        X = np.linspace(-1, 0.3, 500)
        y = np.empty((len(X), 2))
        y[:, 0] = (X**3) + 3 * np.exp(-6 * (X - 0.3) ** 2)
        y[:, 0] += np.random.normal(0, 0.2 * np.abs(X), len(X))
        y[:, 1] = np.log1p(X + 1)
        y[:, 1] += np.log1p(X + 1) * np.random.uniform(size=len(X))

        est = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf, random_state=0)
        est.fit(X.reshape(-1, 1), y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_pred = est.predict(
                X.reshape(-1, 1),
                quantiles=quantiles,
                weighted_quantile=weighted_quantile,
                aggregate_leaves_first=aggregate_leaves_first,
            )
        score = est.score(X.reshape(-1, 1), y, quantiles=0.5)
        assert y_pred.ndim == (3 if isinstance(quantiles, list) else 2)
        assert y_pred.shape[-1] == y.shape[1]
        assert np.any(y_pred[..., 0] != y_pred[..., 1])
        assert score > 0.97

    # Check that specifying `quantiles` overwrites `default_quantiles`.
    est1 = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf, random_state=0)
    est1.fit(X_train, y_train)
    y_pred_1 = est1.predict(X_test, quantiles=0.5)
    est2 = ForestRegressor(
        n_estimators=1,
        default_quantiles=[0.25, 0.5, 0.75],
        max_samples_leaf=max_samples_leaf,
        random_state=0,
    )
    est2.fit(X_train, y_train)
    y_pred_2 = est2.predict(X_test, quantiles=0.5)
    assert_allclose(y_pred_1, y_pred_2)

    # Check that specifying `interpolation` changes outputs.
    est = ForestRegressor(n_estimators=10, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X_train, y_train)
    y_pred_1 = est.predict(X_test, quantiles=0.5, interpolation="linear")
    y_pred_2 = est.predict(X_test, quantiles=0.5, interpolation="nearest")
    assert np.any(y_pred_1 != y_pred_2)

    # Check error if invalid quantiles.
    assert_raises(ValueError, est.predict, X_test, -0.01)
    assert_raises(ValueError, est.predict, X_test, 1.01)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [None, 1])
@pytest.mark.parametrize("quantiles", [None, "mean", 0.5, [0.2, 0.5, 0.8]])
@pytest.mark.parametrize("weighted_quantile", [True, False])
@pytest.mark.parametrize("aggregate_leaves_first", [True, False])
def test_predict_quantiles(
    name,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    check_predict_quantiles(
        name, max_samples_leaf, quantiles, weighted_quantile, aggregate_leaves_first
    )


def check_quantile_ranks_toy(name):
    # Check rank predictions on toy data.
    ForestRegressor = FOREST_REGRESSORS[name]

    # Check predicted ranks on toy sample.
    X = [[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]]
    y = [-1, -1, 0, 1, 1, 2]

    est = ForestRegressor(n_estimators=1, max_samples_leaf=None, bootstrap=False, random_state=0)
    est.fit(X, y)

    expected = [0.75, 0.75, 0.5, 1.0, 1.0, 1.0]
    y_ranks = est.quantile_ranks(X, y, kind="rank")
    assert_array_equal(y_ranks, expected)

    expected = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    y_ranks = est.quantile_ranks(X, y, kind="weak")
    assert_array_equal(y_ranks, expected)

    expected = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    y_ranks = est.quantile_ranks(X, y, kind="strict")
    assert_array_equal(y_ranks, expected)

    expected = [0.5, 0.5, 0.25, 0.75, 0.5, 0.5]
    y_ranks = est.quantile_ranks(X, y, kind="mean")
    assert_array_equal(y_ranks, expected)

    # Check aggregated and unaggregated predicted ranks.
    est = ForestRegressor(n_estimators=2, max_samples_leaf=None, bootstrap=False, random_state=0)
    est.fit(X, y)

    kwargs = {"aggregate_leaves_first": True}

    expected = [0.625, 0.625, 0.375, 0.875, 0.75, 0.75]
    y_ranks = est.quantile_ranks(X, y, kind="rank", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    y_ranks = est.quantile_ranks(X, y, kind="weak", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    y_ranks = est.quantile_ranks(X, y, kind="strict", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [0.5, 0.5, 0.25, 0.75, 0.5, 0.5]
    y_ranks = est.quantile_ranks(X, y, kind="mean", **kwargs)
    assert_allclose(y_ranks, expected)

    kwargs = {"aggregate_leaves_first": False}

    expected = [0.6875, 0.6875, 0.4375, 0.9375, 0.875, 0.875]
    y_ranks = est.quantile_ranks(X, y, kind="rank", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    y_ranks = est.quantile_ranks(X, y, kind="weak", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    y_ranks = est.quantile_ranks(X, y, kind="strict", **kwargs)
    assert_allclose(y_ranks, expected)

    expected = [0.5, 0.5, 0.25, 0.75, 0.5, 0.5]
    y_ranks = est.quantile_ranks(X, y, kind="mean", **kwargs)
    assert_allclose(y_ranks, expected)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks_toy(name):
    check_quantile_ranks_toy(name)


def check_quantile_ranks(name):
    # Check rank predictions.
    ForestRegressor = FOREST_REGRESSORS[name]

    # Check predicted ranks on (semi-)random data.
    x1 = np.random.choice(np.arange(0, 101), size=1000)
    e1_high = 10
    e1 = np.random.uniform(0, e1_high)

    X_train = x1.reshape(-1, 1)
    y_train = np.squeeze(x1 * 2 + e1)

    x2 = np.random.choice(np.arange(0, 101), size=4)

    X_test = x2.reshape(-1, 1)
    y_test = np.squeeze(X_test * 2)

    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    est = ForestRegressor(n_estimators=10, random_state=0)

    est.fit(X_train, y_train)
    y_ranks = est.quantile_ranks(X_test, y_test)

    assert y_ranks.shape == (X_test.shape[0],)
    assert np.all(y_ranks >= 0)
    assert np.all(y_ranks <= 1)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks(name):
    check_quantile_ranks(name)


def check_proximity_counts(name):
    # Check proximity counts.
    ForestRegressor = FOREST_REGRESSORS[name]

    # Check proximity counts on toy sample.
    X = [[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]]
    y = [-1, -1, 0, 1, 1, 2]

    # Check that proximity counts match expected counts without bootstrap.
    est = ForestRegressor(n_estimators=5, max_samples_leaf=None, bootstrap=False, random_state=0)
    est.fit(X, y)

    expected = [
        [(0, 5), (1, 5)],
        [(0, 5), (1, 5)],
        [(2, 5), (3, 5)],
        [(2, 5), (3, 5)],
        [(4, 5)],
        [(5, 5)],
    ]
    proximities = est.proximity_counts(X)
    assert proximities == expected

    # Check proximity counts on shuffled toy sample.
    perm = np.random.permutation(len(y))
    perm_i = np.argsort(perm)
    X = list(np.array(X)[perm])
    y = list(np.array(y)[perm])

    est.fit(X, y)

    expected = [sorted((perm_i[k], v) for k, v in expected[p]) for p in perm]
    proximities = est.proximity_counts(X)
    assert_array_equal(np.array(proximities, dtype=object), np.array(expected, dtype=object))

    # Check proximity counts with `max_proximities` greater than `n_samples`.
    proximities = est.proximity_counts(X, max_proximities=10)
    assert_array_equal([len(p) for p in proximities], [len(e) for e in expected])

    # Check proximity counts with fixed `max_proximities` equal to 1.
    expected = [[e[0]] for e in expected]
    proximities = est.proximity_counts(X, max_proximities=1)
    assert_array_equal([len(p) for p in proximities], [len(e) for e in expected])

    # Check error if `max_proximities` < 1.
    assert_raises(ValueError, est.proximity_counts, X, max_proximities=0)

    # Check error if `max_proximities` is a float.
    assert_raises(ValueError, est.proximity_counts, X, max_proximities=1.5)

    # Check that proximity counts match expected counts without splits.
    est = ForestRegressor(
        n_estimators=1,
        max_samples_leaf=None,
        min_samples_leaf=len(X),
        bootstrap=False,
        random_state=0,
    )
    est.fit(X, y)
    proximities = est.proximity_counts(X)
    proximity_counts = [[count for (_, count) in row] for row in proximities]
    assert np.sum(proximity_counts) == (1 * len(X) * len(X))

    # Check proximity counts on the California Housing Prices dataset.
    est = ForestRegressor(n_estimators=10, max_samples_leaf=None, bootstrap=True, random_state=0)
    est.fit(X_california, y_california)

    # Check that proximity counts match bootstrap counts.
    n_samples = len(X_california)
    proximities = est.proximity_counts(X_california)
    X_leaves = est.apply(X_california)
    bootstrap_indices = np.array(
        [_generate_sample_indices(e.random_state, n_samples, n_samples) for e in est.estimators_]
    )
    for train_idx, train_idx_prox in enumerate(proximities):
        for proximity_idx, proximity_count in train_idx_prox:
            bootstrap_count = 0
            pairs = zip(X_leaves[train_idx], X_leaves[proximity_idx])
            for i, (train_leaf_idx, prox_leaf_idx) in enumerate(pairs):
                if train_leaf_idx == prox_leaf_idx:
                    counts = np.sum(bootstrap_indices[i] == proximity_idx)
                    bootstrap_count += counts
            assert proximity_count == bootstrap_count

    # Check that sorting is correctly applied.
    for i in range(len(proximities)):
        for j in range(len(proximities[i]) - 1):
            assert proximities[i][j][1] >= proximities[i][j + 1][1]
    proximities_unsorted = est.proximity_counts(X_california, return_sorted=False)
    assert proximities != proximities_unsorted


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_proximity_counts(name):
    check_proximity_counts(name)


def check_max_samples_leaf(name):
    # Check that the `max_samples_leaf` parameter correctly samples leaves.
    X = X_california
    y = y_california

    ForestRegressor = FOREST_REGRESSORS[name]

    max_leaf_sizes = []
    for max_samples_leaf in [0.99 / len(X), 1, 3.0 / len(X), 5, 20, None]:
        est = ForestRegressor(n_estimators=10, max_samples_leaf=max_samples_leaf, random_state=0)
        est.fit(X, y)

        max_leaf_size = 0
        for _, tree_lookup in enumerate(est._get_y_train_leaves(X, 1)):
            for leaf_samples in np.squeeze(tree_lookup, -2):
                n_leaf_samples = len([x for x in leaf_samples if x != 0])
                if n_leaf_samples > max_leaf_size:
                    max_leaf_size = n_leaf_samples

                # Check that ints and floats correctly sample leaves.
                if isinstance(max_samples_leaf, int):
                    assert n_leaf_samples <= max_samples_leaf
                elif isinstance(max_samples_leaf, float):
                    max_int = math.ceil(max_samples_leaf * len(X))
                    assert n_leaf_samples <= max_int

        max_leaf_sizes.append(max_leaf_size)

    # Check that larger values do not result in smaller maximum leaf sizes.
    for max_1, max_2 in zip(max_leaf_sizes[::], max_leaf_sizes[1::]):
        assert max_1 <= max_2

    # Check error if `max_samples_leaf` <= 0, float larger than 1, or string.
    for max_samples_leaf in [0, 1.5, "None"]:
        for param_validation in [True, False]:
            est = ForestRegressor(n_estimators=1, max_samples_leaf=max_samples_leaf)
            est.param_validation = param_validation
            assert_raises(ValueError, est.fit, X, y)
            est.max_samples_leaf = max_samples_leaf
            assert_raises(ValueError, est._get_y_train_leaves, X, 1)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_max_samples_leaf(name):
    check_max_samples_leaf(name)


def check_oob_samples(name):
    # Check OOB sample generation.
    X = X_california
    y = y_california

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, bootstrap=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, X_indices = est.fit(X, y)._oob_samples(X)

    assert len(X_indices) == len(y)

    mean_oob_count = X_indices[X_indices == 1].sum() / X_indices.shape[1]
    actual = mean_oob_count / len(y)
    expected = 1.0 / math.e  # OOB count should asymptotically approach 1/e
    assert_almost_equal(actual, expected, decimal=2)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_oob_samples(name):
    check_oob_samples(name)


def check_oob_samples_duplicates(name):
    # Check OOB sampling with duplicates.
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [13, 14, 15],
        ]
    )
    y = np.array([-1, 10, 20, 30, 40, 41], dtype=np.float64)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=1, bootstrap=True, oob_score=True, random_state=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        est.fit(X, y)
        _, X_indices1 = est._oob_samples(X, duplicates=None)
        _, X_indices2 = est._oob_samples(X, duplicates=[[4, 5]])
    assert len(X_indices1) == len(y)
    assert len(X_indices2) == len(y)
    assert np.all(X_indices1[:4] == X_indices2[:4])
    assert X_indices2[4] == X_indices2[5]


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_oob_samples_duplicates(name):
    check_oob_samples_duplicates(name)


def check_predict_oob(
    name,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    # Check OOB predictions.
    X = X_california
    y = y_california

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=20,
        max_samples_leaf=max_samples_leaf,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    )
    est.fit(X, y)

    n_quantiles = None
    median_idx = None
    if isinstance(quantiles, list):
        n_quantiles = len(quantiles)
        median_idx = quantiles.index(0.5)

    # Check that `R^2` score from OOB predictions is close to `oob_score_`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = est.predict(
            X,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )
    y_pred_score = r2_score(y, y_pred[:, median_idx])
    if n_quantiles is not None:
        assert y_pred.shape == (len(X), n_quantiles)
    else:
        assert y_pred.shape == (len(X),)
    if quantiles == "mean" and aggregate_leaves_first is False:
        assert est.oob_score_ == y_pred_score
    else:
        assert abs(est.oob_score_ - y_pred_score) < 0.1

    # Check OOB predictions on chunks of (permuted) indexed samples.
    if n_quantiles is not None:
        y_pred_chunks = np.empty((len(X), n_quantiles))
    else:
        y_pred_chunks = np.empty((len(X),))
    perm = np.random.permutation(len(X))
    for indices in np.split(np.arange(len(X)), range(100, len(X), 100)):
        X_chunk = X[perm[indices]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_pred_chunk = est.predict(
                X_chunk,
                quantiles=quantiles,
                weighted_quantile=weighted_quantile,
                aggregate_leaves_first=aggregate_leaves_first,
                oob_score=True,
                indices=perm[indices],
            )
        y_pred_chunks[indices, ...] = y_pred_chunk
        if n_quantiles is not None:
            assert y_pred_chunk.shape == (len(X_chunk), n_quantiles)
        else:
            assert y_pred_chunk.shape == (len(X_chunk),)
    y_pred_chunks = y_pred_chunks[np.argsort(perm)]

    # Check that chunked OOB predictions equal non-chunked OOB predictions.
    assert np.all(y_pred == y_pred_chunks)

    # Check precomputed unsampled indices for OOB scoring.
    unsampled_indices = [None] * est.n_estimators
    for i, estimator in enumerate(est.estimators_):
        unsampled_indices[i] = est._get_unsampled_indices(estimator)
    est.unsampled_indices_ = unsampled_indices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_precomputed_indices = est.predict(
            X,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )
    assert np.all(y_pred == y_pred_precomputed_indices)

    # Check single-row OOB scoring.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_single_row = est.predict(
            X[:1],
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
            indices=np.zeros(1),
        )
    assert np.all(y_pred[:1] == y_pred_single_row)

    # Check that OOB predictions indexed by -1 return IB predictions.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_ib = est.predict(
            X,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=False,
        )
        y_pred_oob = est.predict(
            X,
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
            indices=-np.ones(len(X)),
        )
    assert np.all(y_pred_ib == y_pred_oob)

    # Check OOB predictions with `default_quantiles`.
    est1 = ForestRegressor(n_estimators=1, random_state=0)
    est1.fit(X, y)
    est2 = ForestRegressor(n_estimators=1, default_quantiles=quantiles, random_state=0)
    est2.fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred_oob1 = est1.predict(X, quantiles=quantiles)
        y_pred_oob2 = est2.predict(X)
    assert_allclose(y_pred_oob1, y_pred_oob2)

    # Check error if OOB score without `indices` do not match training count.
    assert_raises(ValueError, est.predict, X[:1], oob_score=True)

    # Check error if OOB score with `indices` do not match samples count.
    assert_raises(
        ValueError,
        est.predict,
        X,
        oob_score=True,
        indices=-np.ones(len(X) - 1),
    )

    # Check warning if not enough estimators.
    with np.errstate(divide="ignore", invalid="ignore"):
        est = ForestRegressor(n_estimators=4, bootstrap=True, oob_score=True, random_state=0)
        with pytest.warns(UserWarning):
            est.fit(X, y)
            est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=weighted_quantile,
                aggregate_leaves_first=aggregate_leaves_first,
                oob_score=True,
            )
            y_pred = est.predict(
                X,
                quantiles=quantiles,
                weighted_quantile=weighted_quantile,
                aggregate_leaves_first=aggregate_leaves_first,
                oob_score=True,
            )
            assert np.isnan(y_pred).any()

    # Check error if no bootstrapping.
    est = ForestRegressor(n_estimators=1, bootstrap=False)
    est.fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_raises(
            ValueError,
            est.predict,
            X,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )
        assert np.all(est._get_unsampled_indices(est.estimators_[0]) == np.array([]))

    # Check error if number of scoring and training samples are different.
    est = ForestRegressor(n_estimators=1, bootstrap=True)
    est.fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_raises(
            ValueError,
            est.predict,
            X[:1],
            y[:1],
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [None, 1])
@pytest.mark.parametrize("quantiles", [None, "mean", 0.5, [0.2, 0.5, 0.8]])
@pytest.mark.parametrize("weighted_quantile", [True, False])
@pytest.mark.parametrize("aggregate_leaves_first", [True, False])
def test_predict_oob(
    name,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    check_predict_oob(
        name,
        max_samples_leaf,
        quantiles,
        weighted_quantile,
        aggregate_leaves_first,
    )


def check_quantile_ranks_oob(name):
    # Check OOB quantile rank predictions.
    X = X_california
    y = y_california

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=20, bootstrap=True, oob_score=True, random_state=0)
    est.fit(X, y)

    y_ranks = est.quantile_ranks(X, y, oob_score=True)
    assert len(y_ranks) == len(y)
    assert np.all(y_ranks >= 0)
    assert np.all(y_ranks <= 1)

    # Check OOB predicted ranks on chunks of indexed samples.
    y_ranks_chunks = np.empty(len(X))
    for indices in np.split(np.arange(len(X)), range(100, len(X), 100)):
        X_chunk, y_chunk = X[indices], y[indices]
        y_ranks_chunk = est.quantile_ranks(X_chunk, y_chunk, oob_score=True, indices=indices)
        y_ranks_chunks[indices] = y_ranks_chunk
        assert len(y_ranks_chunk) == len(X_chunk)

    # Check that chunked OOB ranks equal non-chunked OOB ranks.
    assert np.all(y_ranks_chunks >= 0)
    assert np.all(y_ranks_chunks <= 1)
    assert np.all(y_ranks == y_ranks_chunks)

    # Check that OOB ranks indexed by -1 return IB ranks.
    y_ranks_ib = est.quantile_ranks(X, y, oob_score=False)
    y_ranks_oob = est.quantile_ranks(X, y, oob_score=True, indices=-np.ones(len(X)))
    assert np.all(y_ranks_ib == y_ranks_oob)

    # Check warning if not enough estimators.
    with np.errstate(divide="ignore", invalid="ignore"):
        est = ForestRegressor(n_estimators=4, bootstrap=True, oob_score=True, random_state=0)
        with pytest.warns(UserWarning):
            est.fit(X, y)
            y_ranks = est.quantile_ranks(X, y, oob_score=True)
            assert np.isnan(y_ranks).any()

    # Check error if no bootstrapping.
    est = ForestRegressor(n_estimators=1, bootstrap=False)
    est.fit(X, y)
    assert_raises(ValueError, est.quantile_ranks, X, y, oob_score=True)

    # Check error if number of scoring and training samples are different.
    est = ForestRegressor(n_estimators=1, bootstrap=True)
    est.fit(X, y)
    assert_raises(ValueError, est.quantile_ranks, X[:1], y[:1], oob_score=True)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks_oob(name):
    check_quantile_ranks_oob(name)


def check_proximity_counts_oob(name):
    # Check OOB proximity counts.
    X = X_california
    y = y_california

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=20,
        max_samples_leaf=None,
        bootstrap=True,
        oob_score=True,
        random_state=0,
    )
    est.fit(X, y)

    proximities = est.proximity_counts(X, oob_score=True)

    # Check that OOB proximity counts match OOB bootstrap counts.
    X_leaves = est.apply(X)
    n_samples = len(X)
    bootstrap_indices = np.array(
        [_generate_sample_indices(e.random_state, n_samples, n_samples) for e in est.estimators_]
    )
    for train_idx, train_idx_prox in enumerate(proximities):
        for proximity_idx, proximity_count in train_idx_prox:
            bootstrap_count = 0
            pairs = zip(X_leaves[train_idx], X_leaves[proximity_idx])
            for i, (train_leaf_idx, prox_leaf_idx) in enumerate(pairs):
                if train_idx in bootstrap_indices[i]:
                    continue  # skip IB trees
                if train_leaf_idx == prox_leaf_idx:
                    counts = np.sum(bootstrap_indices[i] == proximity_idx)
                    bootstrap_count += counts
            assert proximity_count == bootstrap_count

    # Check OOB proximity counts on chunks of indexed samples.
    proximities_chunks = [None] * len(X)
    for indices in np.split(np.arange(len(X)), range(100, len(X), 100)):
        X_chunk = X[indices]
        proximities_chunk = est.proximity_counts(X_chunk, oob_score=True, indices=indices)
        for i, idx in enumerate(indices):
            proximities_chunks[idx] = proximities_chunk[i]
        assert len(proximities_chunk) == len(X_chunk)

    # Check that chunked OOB proximity counts equal non-chunked OOB counts.
    assert all(x == y for x, y in zip(proximities, proximities_chunks))

    # Check that OOB proximity counts indexed by -1 return IB counts.
    proximities_ib = est.proximity_counts(X, oob_score=False)
    proximities_oob = est.proximity_counts(X, oob_score=True, indices=-np.ones(len(X)))
    assert np.all(proximities_ib == proximities_oob)

    # Check warning if not enough estimators.
    with np.errstate(divide="ignore", invalid="ignore"):
        est = ForestRegressor(
            n_estimators=4,
            max_samples_leaf=None,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        )
        with pytest.warns(UserWarning):
            est.fit(X, y)
            proximities = est.proximity_counts(X, oob_score=True)
            for train_idx, train_idx_prox in enumerate(proximities):
                assert any(len(x) == 0 for x in proximities)

    # Check error if no bootstrapping.
    est = ForestRegressor(n_estimators=1, max_samples_leaf=None, bootstrap=False)
    est.fit(X, y)
    assert_raises(ValueError, est.proximity_counts, X, oob_score=True)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_proximity_counts_oob(name):
    check_proximity_counts_oob(name)


def test_calc_quantile():
    # Check quantile calculations.
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    interpolations = [b"linear", b"lower", b"higher", b"midpoint", b"nearest"]

    inputs = [
        [1],
        [2],
        [2, 3],
        [2, 3, 4],
        [1, 1, 1, 1, 1],
        [1, 2],
        [1, 2, 2, 2, 2, 2],
        [1, 2, 3],
        [1, 2, 3, 3, 3, 3, 3],
        [1, 1, 2],
        [15, 20, 35, 40, 50],
        [3, 20, 20, 1, 1, 1, 40],
    ]

    # Check that output for non-empty lists is equal to ``np.quantile``.
    for interpolation in interpolations:
        kwargs1 = {"interpolation": interpolation}
        kwargs2 = {"method": interpolation.decode()}
        for i in inputs:
            actual = calc_quantile(i, quantiles, **kwargs1)
            expected = [np.quantile(i, q, **kwargs2) for q in quantiles]
            assert_allclose(actual, expected)

    # Check that linear interpolation at 0.5 quantile equals median.
    for i in inputs:
        actual = calc_quantile(i, [0.5], interpolation=b"linear")
        expected = [np.median(i)]
        assert_allclose(actual, expected)

    # Check mean.
    for i in inputs:
        actual = calc_mean(i)
        expected = [np.mean(i)]
        assert_allclose(actual, expected)

    # Check that quantile order is respected.
    for i in inputs:
        for q in [quantiles, quantiles[::-1]]:
            actual = calc_quantile(i, q)
            for idx in range(len(quantiles) - 1):
                if q[idx] <= q[idx + 1]:
                    assert np.all(np.less_equal(actual[idx], actual[idx + 1]))
                else:
                    assert np.all(np.less_equal(actual[idx + 1], actual[idx]))

    inputs = []

    # Check that empty array is returned for empty list.
    actual = calc_quantile(inputs, quantiles)
    expected = []
    assert_array_equal(actual, expected)

    inputs = [2, 4, 3]

    # Check that sorting is correctly applied.
    actual1 = calc_quantile(inputs, quantiles, issorted=True)
    actual2 = calc_quantile(inputs, quantiles, issorted=False)
    assert actual1 != actual2

    # Check error if invalid parameters.
    assert_raises(TypeError, calc_quantile, [1, 2], 0.5)
    assert_raises(TypeError, calc_quantile, [1, 2], [0.5], interpolation=None)


def test_calc_weighted_quantile():
    # Check weighted quantile calculations.
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    interpolations = [b"linear", b"lower", b"higher", b"midpoint", b"nearest"]

    def _dicts_to_weighted_inputs(input_dicts):
        """Convert dicts to lists of inputs and weights."""
        for d in input_dicts:
            inputs = []
            weights = []
            for k, v in d.items():
                inputs.extend([k])
                weights.extend([v])
            yield inputs, weights

    def _dicts_to_unweighted_inputs(input_dicts):
        """Convert dicts to expanded lists of inputs."""
        for d in input_dicts:
            inputs = []
            for k, v in d.items():
                if v > 0:
                    inputs.extend([k] * v)
            yield inputs

    def _dicts_to_input_pairs(input_dicts):
        """Convert dicts to pairs of weighted and unweighted inputs."""
        weighted = _dicts_to_weighted_inputs(input_dicts)
        unweighted = _dicts_to_unweighted_inputs(input_dicts)
        return zip(weighted, unweighted)

    # Dicts of inputs and frequency counts (weights).
    inputs = [
        {1: 1},
        {2: 1, 3: 0},
        {2: 1, 3: 1},
        {2: 1, 3: 1, 4: 1},
        {1: 5, 2: 0, 3: 0},
        {1: 1, 2: 1},
        {1: 1, 2: 5},
        {1: 1, 2: 1, 3: 1},
        {1: 1, 2: 1, 3: 5},
        {1: 2, 2: 1, 3: 0},
        {15: 1, 20: 1, 35: 1, 40: 1, 50: 1},
        {3: 1, 20: 2, 1: 3, 40: 1},
    ]

    # Check that output for non-empty dicts is equal to ``np.quantile``.
    for interpolation in interpolations:
        kwargs1 = {"interpolation": interpolation}
        kwargs2 = {"method": interpolation.decode()}
        for (i1, w1), i2 in _dicts_to_input_pairs(inputs):
            actual = calc_weighted_quantile(i1, w1, quantiles, **kwargs1)
            expected = [np.quantile(i2, q, **kwargs2) for q in quantiles]
            assert_allclose(actual, expected)

    # Check that linear interpolation at 0.5 quantiles equals median.
    for (i1, w1), i2 in _dicts_to_input_pairs(inputs):
        actual = calc_weighted_quantile(i1, w1, [0.5], interpolation=b"linear")
        expected = [np.median(i2)]
        assert_allclose(actual, expected)

    # Check mean.
    for (i1, w1), i2 in _dicts_to_input_pairs(inputs):
        actual = calc_weighted_mean(i1, w1)
        expected = [np.mean(i2)]
        assert_allclose(actual, expected)

    # Check that quantile order is respected.
    for i1, w1 in _dicts_to_weighted_inputs(inputs):
        for q in [quantiles, quantiles[::-1]]:
            actual = calc_weighted_quantile(i1, w1, q)
            for idx in range(len(quantiles) - 1):
                if q[idx] <= q[idx + 1]:
                    assert np.all(np.less_equal(actual[idx], actual[idx + 1]))
                else:
                    assert np.all(np.less_equal(actual[idx + 1], actual[idx]))

    inputs = [1, 2, 3, 3, 3, 3, 3, 3, 4, 5]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Check that repeat inputs are calculated correctly.
    actual = calc_weighted_quantile(inputs, weights, quantiles)
    expected = [np.quantile(inputs, q) for q in quantiles]
    assert_allclose(actual, expected)

    inputs = [
        {1: 0},
        {2: 0},
        {1: 0, 2: 0},
    ]

    # Check that empty array is returned for weights that sum to 0.
    actual = [
        calc_weighted_quantile(i, w, quantiles) for i, w in _dicts_to_weighted_inputs(inputs)
    ]
    expected = [[], [], []]
    assert_array_equal(actual, expected)

    inputs = np.array([], dtype=np.float64)
    weights = np.array([], dtype=np.int32)

    # Check that empty array is returned for empty lists.
    actual = calc_weighted_quantile(inputs, weights, quantiles)
    expected = []
    assert_array_equal(actual, expected)

    inputs = [2, 4, 3]
    weights = [1, 1, 1]

    # Check that sorting is correctly applied.
    actual1 = calc_weighted_quantile(inputs, weights, quantiles, issorted=True)
    actual2 = calc_weighted_quantile(inputs, weights, quantiles, issorted=False)
    assert actual1 != actual2

    # Check error if invalid parameters.
    assert_raises(TypeError, calc_weighted_quantile, [1, 2], [1, 1], 0.5)
    assert_raises(
        TypeError,
        calc_weighted_quantile,
        [1, 2],
        [1, 1],
        [0.5],
        interpolation=None,
    )


def test_calc_quantile_rank():
    # Check quantile rank calculations.
    kinds = [b"rank", b"weak", b"strict", b"mean"]

    inputs = [
        [1],
        [2],
        [2, 3],
        [2, 3, 4],
        [1, 1, 1, 1, 1],
        [1, 2],
        [1, 2, 2, 2, 2, 2],
        [1, 2, 3],
        [1, 2, 3, 3, 3, 3, 3],
        [1, 1, 2],
        [15, 20, 35, 40, 50],
        [3, 20, 20, 1, 1, 1, 40],
    ]
    scores = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.float64)

    # Check that output for non-empty lists is equal to ``percentileofscore``.
    for kind in kinds:
        kwargs1 = {"kind": kind}
        kwargs2 = {"kind": kind.decode()}
        for i, s in zip(inputs, scores):
            actual = calc_quantile_rank(i, s, **kwargs1)
            expected = percentileofscore(i, s, **kwargs2) / 100.0
            assert_allclose(actual, expected)

    inputs = []
    scores = np.array([1], dtype=np.float64)

    # Check that -1 is returned for empty list.
    actual = calc_quantile_rank(inputs, scores)
    expected = [-1]
    assert_array_equal(actual, expected)

    inputs = [2, 4, 3]
    scores = np.array([3], dtype=np.float64)

    # Check that sorting is correctly applied.
    actual1 = calc_quantile_rank(inputs, scores, issorted=True)
    actual2 = calc_quantile_rank(inputs, scores, issorted=False)
    assert actual1 != actual2

    # Check error if invalid parameters.
    assert_raises(TypeError, calc_quantile_rank, [1, 2], [1])
    assert_raises(
        TypeError,
        calc_quantile_rank,
        [1, 2],
        np.array([1], dtype=np.float64),
        kind=None,
    )


def test_generate_unsampled_indices():
    # Check unsampled indices generation.
    max_index = 20
    duplicates = [[1, 4], [19, 10], [2, 3, 5], [6, 13]]

    def _generate_unsampled_indices(sample_indices):
        return generate_unsampled_indices(
            np.array(sample_indices, dtype=np.int64),
            duplicates=duplicates,
        )

    # If all indices are sampled, there are no unsampled indices.
    indices = [idx for idx in range(max_index)]
    expected = np.array([], dtype=np.int64)
    assert_array_equal(_generate_unsampled_indices(indices), expected)

    # Index 7 has no duplicates, and thus should be the only unsampled index.
    indices = [7 for _ in range(max_index)]
    expected = np.array([idx for idx in range(max_index) if idx != 7])
    assert_array_equal(_generate_unsampled_indices(indices), expected)

    # Check sample indices [0, 1, 2] with duplicates set(1, 4) + set(2, 3, 5),
    # which excludes [0, 1, 2, 3, 4, 5] (i.e., range(6)) from unsampled.
    indices = [idx % 3 for idx in range(max_index)]
    expected = [x for x in range(max_index) if x not in range(6)]
    assert_array_equal(_generate_unsampled_indices(indices), expected)
