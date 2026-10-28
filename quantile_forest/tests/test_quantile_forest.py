"""Testing for the quantile forest module (quantile_forest._quantile_forest)."""

import math
import pickle
import random
import warnings
from typing import Any

import joblib
import numpy as np
import pytest
import sklearn
from scipy import sparse
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
)
from sklearn.utils.estimator_checks import check_sample_weights_not_an_array
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted

from quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor
from quantile_forest._quantile_forest import _hash_X, _nanquantile_rows
from quantile_forest._quantile_forest_fast import (
    calc_mean,
    calc_quantile,
    calc_quantile_rank,
    calc_weighted_mean,
    calc_weighted_quantile,
)

sklearn_version = parse_version(sklearn.__version__)

np.random.seed(0)

FOREST_REGRESSORS: dict[str, Any] = {
    "ExtraTreesQuantileRegressor": ExtraTreesQuantileRegressor,
    "RandomForestQuantileRegressor": RandomForestQuantileRegressor,
}


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, 3, None])
def test_fit_preserves_python_random_state(name, max_samples_leaf):
    """Leaf subsampling must not reseed the caller's Python random generator."""
    original_state = random.getstate()
    try:
        random.seed(42)
        expected_state = random.getstate()
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20, dtype=float)
        FOREST_REGRESSORS[name](
            n_estimators=3,
            max_depth=1,
            max_samples_leaf=max_samples_leaf,
            random_state=0,
        ).fit(X, y)
        assert random.getstate() == expected_state
    finally:
        random.setstate(original_state)


def test_leaf_subsampling_keeps_seeded_selection():
    """Use the same seeded shuffle while keeping the caller's state intact."""
    indices = list(range(10))
    random.Random(17).shuffle(indices)
    expected = np.asarray(indices[:3]) + 1
    original_state = random.getstate()
    try:
        actual = RandomForestQuantileRegressor._get_y_train_leaves_slice(
            bootstrap_indices=np.arange(1, 11).reshape(-1, 1),
            X_leaves_bootstrap=np.ones(10, dtype=np.intp),
            sample_weight=None,
            leaf_subsample=True,
            max_node_count=2,
            max_samples_leaf=3,
            random_state=17,
        )
        assert_array_equal(actual[1, 0], expected)
        assert random.getstate() == original_state
    finally:
        random.setstate(original_state)


def check_regression_toy(name, weighted_quantile):
    """Check quantile regression on a toy dataset."""
    quantiles = [0.25, 0.5, 0.75]

    # Toy sample.
    X = [[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]]
    y = [-1, -1, 0, 1, 1, 2]
    y_test = [[-1, -1], [2, 2], [3, 2]]

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=10, max_samples_leaf=None, bootstrap=False, random_state=0)
    est.fit(X, y, sample_weight=np.ones(len(X)))

    # Check model and apply outputs shape.
    leaf_indices = est.apply(X)
    assert leaf_indices.shape == (len(X), est.n_estimators)
    assert 10 == len(est)

    # Check aggregated quantile predictions.
    y_true = [[0.0, 0.5, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    y_pred = est.predict(
        y_test,
        quantiles=quantiles,
        weighted_quantile=weighted_quantile,
        aggregate_leaves_first=True,
    )
    assert_allclose(y_pred, y_true)

    # Check unaggregated quantile predictions.
    y_true = [[0.25, 0.5, 0.75], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    y_pred = est.predict(
        y_test,
        quantiles=quantiles,
        weighted_quantile=weighted_quantile,
        aggregate_leaves_first=False,
    )
    assert_allclose(y_pred, y_true)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("weighted_quantile", [True, False])
def test_regression_toy(name, weighted_quantile):
    check_regression_toy(name, weighted_quantile)


def check_regression_params(name):
    """Check quantile regression model parameters."""
    params = {
        "criterion": "squared_error",
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": 16,
        "min_impurity_decrease": 0.0,
        "ccp_alpha": 0.0,
        "monotonic_cst": [0, 1, -1, 0],
    }

    ForestRegressor = FOREST_REGRESSORS[name]

    X, y = datasets.make_regression(n_features=4, n_informative=2, shuffle=True, random_state=0)

    est = ForestRegressor(**params, random_state=0).fit(X, y)

    for param in params:
        assert getattr(est, param) == getattr(est.estimators_[0], param)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_regression_params(name):
    check_regression_params(name)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("weighted_leaves", [False, True])
@pytest.mark.parametrize("aggregate_leaves_first", [True, False])
def test_predict_weighted_rows_independent(name, weighted_leaves, aggregate_leaves_first):
    """Check that the weighted prediction of a sample doesn't depend on other samples."""
    X, y = datasets.make_regression(n_samples=200, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, max_samples_leaf=None, bootstrap=True, random_state=0)
    est.fit(X, y)

    kwargs = {
        "weighted_quantile": True,
        "weighted_leaves": weighted_leaves,
        "aggregate_leaves_first": aggregate_leaves_first,
    }
    for quantiles in [[0.25, 0.5, 0.75], "mean"]:
        y_pred = est.predict(X[:20], quantiles, **kwargs)
        for i in range(20):
            assert_array_equal(est.predict(X[[i]], quantiles, **kwargs)[0], y_pred[i])
        assert_array_equal(est.predict(X[19::-1], quantiles, **kwargs), y_pred[::-1])

    proximities = est.proximity_counts(X[:20])
    for i in range(20):
        assert est.proximity_counts(X[[i]]) == proximities[i : i + 1]


def check_predict_quantiles_toy(name):
    """Check quantile predictions on toy data."""
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
        with pytest.raises(AssertionError):
            assert_allclose(y_pred1, y_pred2)

        # Check that leaf weighting without weighted quantiles does nothing.
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


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
def test_predict_quantiles_array_like(name, max_samples_leaf):
    """Check that quantiles can be any array-like of floats, or "mean" in a list."""
    X, y = datasets.make_regression(n_samples=100, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X, y)

    quantiles = [0.25, 0.5, 0.75]
    y_pred = est.predict(X, quantiles=quantiles)
    for quantiles_array_like in [
        tuple(quantiles),
        np.array(quantiles),
        np.array(quantiles, dtype=np.float32),
    ]:
        assert_allclose(est.predict(X, quantiles=quantiles_array_like), y_pred)

    for quantile in [np.float32(0.5), np.array(0.5), np.int64(1)]:
        assert_allclose(est.predict(X, quantiles=quantile), est.predict(X, float(quantile)))

    est.set_params(default_quantiles=np.array(quantiles))
    assert_allclose(est.predict(X), y_pred)

    assert_allclose(est.predict(X, quantiles=["mean"]), est.predict(X, quantiles="mean"))


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
@pytest.mark.parametrize("weighted_quantile", [True, False])
def test_predict_quantiles_invalid(name, max_samples_leaf, weighted_quantile):
    """Check that invalid quantiles raise an error instead of predicting."""
    X, y = datasets.make_regression(n_samples=100, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, max_samples_leaf=max_samples_leaf, random_state=0)
    est.fit(X, y)

    for quantiles in [
        np.nan,
        [0.5, np.nan],
        [],
        -0.01,
        1.01,
        "median",
        [0.5, "mean"],
        [[0.5]],
        [0.1, [0.2, 0.3]],  # ragged
    ]:
        with pytest.raises(ValueError, match="Quantiles must"):
            est.predict(X, quantiles=quantiles, weighted_quantile=weighted_quantile)

    # The Cython predictor checks its inputs too.
    X_leaves = est.apply(X)
    for quantiles in [[np.nan], []]:
        with pytest.raises(ValueError, match="Quantiles must"):
            est.forest_.predict(quantiles, X_leaves, weighted_quantile=weighted_quantile)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
def test_predict_uses_fitted_trees(name, max_samples_leaf):
    """Check that predictions use the fitted trees, even if n_estimators changes."""
    X, y = datasets.make_regression(n_samples=120, n_features=2, random_state=0)
    X_train, y_train, X_test = X[:100], y[:100], X[100:]

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=10, max_samples_leaf=max_samples_leaf, bootstrap=True, random_state=0
    )
    est.fit(X_train, y_train)

    quantiles = [0.1, 0.5, 0.9]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some samples have no OOB trees
        y_pred = est.predict(X_test, quantiles=quantiles)
        y_pred_oob = est.predict(X_train, quantiles=quantiles, oob_score=True)

        for n_estimators in [5, 20]:  # e.g., set before a warm-start refit
            est.set_params(n_estimators=n_estimators)
            assert_allclose(est.predict(X_test, quantiles=quantiles), y_pred)
            assert_allclose(est.predict(X_train, quantiles=quantiles, oob_score=True), y_pred_oob)


def check_predict_quantiles(
    name,
    max_samples,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    """Check quantile predictions."""
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

    est = ForestRegressor(
        n_estimators=10,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est.fit(X_train, y_train)
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

    # Check predicted quantiles on regression dataset.
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    est = ForestRegressor(
        n_estimators=10,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est.fit(X_train, y_train)
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
    est = ForestRegressor(
        n_estimators=10,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
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
    est = ForestRegressor(
        n_estimators=1,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est.fit(X_train, y_train)
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
    est = ForestRegressor(
        n_estimators=1,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
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
    est = ForestRegressor(
        n_estimators=1,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
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
        else:
            est1 = RandomForestRegressor(
                n_estimators=1 if max_samples_leaf == 1 else 10,
                random_state=0,
            )
        est2 = ForestRegressor(
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

        est = ForestRegressor(
            n_estimators=1,
            max_samples_leaf=max_samples_leaf,
            max_samples=max_samples,
            bootstrap=True,
            random_state=0,
        )
        est.fit(X.reshape(-1, 1), y)
        y_pred = est.predict(
            X.reshape(-1, 1),
            quantiles=quantiles,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
        )
        score = est.score(X.reshape(-1, 1), y, quantiles=0.5)
        assert y_pred.ndim == (3 if isinstance(quantiles, list) else 2)
        assert y_pred.shape[1] == y.shape[1]
        assert np.any(y_pred[:, 0, ...] != y_pred[:, 1, ...])
        assert score > 0.95

        # Check unaggregated predictions with absolute error criterion.
        if quantiles == 0.5:
            X_train_mae = np.array([[1], [1], [3], [1]])
            y_train_mae = np.arange(4)

            X_test_mae = np.array([[5], [4], [2], [0]])

            params = {"criterion": "absolute_error", "max_depth": 1, "random_state": 0}

            if name == "ExtraTreesQuantileRegressor":
                est1 = ExtraTreesRegressor(**params)
            else:
                est1 = RandomForestRegressor(**params)
            est2 = ForestRegressor(max_samples_leaf=None, **params)

            est1.fit(X_train_mae, y_train_mae)
            est2.fit(X_train_mae, y_train_mae)

            y_pred1 = est1.predict(X_test_mae)
            y_pred2 = est2.predict(X_test_mae, quantiles=0.5, aggregate_leaves_first=False)
            assert_allclose(y_pred1, y_pred2)

    # Check that specifying `quantiles` overwrites `default_quantiles`.
    est1 = ForestRegressor(
        n_estimators=1,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est1.fit(X_train, y_train)
    y_pred_1 = est1.predict(X_test, quantiles=0.5)
    est2 = ForestRegressor(
        n_estimators=1,
        default_quantiles=[0.25, 0.5, 0.75],
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est2.fit(X_train, y_train)
    y_pred_2 = est2.predict(X_test, quantiles=0.5)
    assert_allclose(y_pred_1, y_pred_2)

    # Check that specifying `interpolation` changes outputs.
    est = ForestRegressor(
        n_estimators=10,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
        bootstrap=True,
        random_state=0,
    )
    est.fit(X_train, y_train)
    y_pred_1 = est.predict(X_test, quantiles=0.5, interpolation="linear")
    y_pred_2 = est.predict(X_test, quantiles=0.5, interpolation="nearest")
    assert np.any(y_pred_1 != y_pred_2)

    # Check error if invalid quantiles.
    with pytest.raises(ValueError):
        est.predict(X_test, -0.01)
    with pytest.raises(ValueError):
        est.predict(X_test, 1.01)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples", [None, 0.5, 1.0, 100])
@pytest.mark.parametrize("max_samples_leaf", [None, 1])
@pytest.mark.parametrize("quantiles", [None, "mean", 0.5, [0.2, 0.5, 0.8]])
@pytest.mark.parametrize("weighted_quantile", [True, False])
@pytest.mark.parametrize("aggregate_leaves_first", [True, False])
def test_predict_quantiles(
    name,
    max_samples,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    check_predict_quantiles(
        name,
        max_samples,
        max_samples_leaf,
        quantiles,
        weighted_quantile,
        aggregate_leaves_first,
    )


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
@pytest.mark.parametrize("sparse_container", [sparse.csr_matrix, sparse.csc_matrix])
def test_predict_sparse(name, max_samples_leaf, sparse_container):
    """Check that sparse inputs give the same predictions as dense inputs."""
    X, y = datasets.make_regression(n_samples=200, n_features=4, random_state=0)
    X[X < 0.5] = 0
    X_sparse = sparse_container(X)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=10, max_samples_leaf=max_samples_leaf, bootstrap=True, random_state=0
    )
    est.fit(X_sparse, y)

    for quantiles in [[0.1, 0.5, 0.9], "mean"]:
        assert_allclose(est.predict(X_sparse, quantiles), est.predict(X, quantiles))
    assert_allclose(est.score(X_sparse, y), est.score(X, y))

    # OOB predictions on the sparse training data don't warn that the data differ.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        y_pred_oob = est.predict(X_sparse, quantiles=0.5, oob_score=True)
    assert not any("not identical" in str(w.message) for w in record)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dense X is not in the training format
        assert_allclose(y_pred_oob, est.predict(X, quantiles=0.5, oob_score=True))


def test_hash_X_sparse():
    """Check that sparse inputs hash equally if they hold the same matrix."""
    X = np.array([[0, 1.5, 0], [2, 0, 0], [0, 0.5, 3]], dtype=np.float32)
    X_csc = sparse.csc_matrix(X)

    # The same matrix, with unsorted indices in column 1 and a duplicate entry in column 2.
    data = np.array([2, 0.5, 1.5, 1, 2], dtype=np.float32)
    indices = np.array([1, 2, 0, 2, 2], dtype=np.int32)
    indptr = np.array([0, 1, 3, 5], dtype=np.int32)
    X_noncanonical = sparse.csc_matrix((data, indices, indptr), shape=X.shape)
    assert not X_noncanonical.has_canonical_format
    assert_array_equal(X_noncanonical.toarray(), X)

    for X_same in [X_noncanonical, X_csc.tocsr()]:
        assert _hash_X(X_same) == _hash_X(X_csc)
    assert _hash_X(sparse.csc_matrix(2 * X)) != _hash_X(X_csc)

    # Hashing does not modify the input.
    assert_array_equal(X_noncanonical.indices, [1, 2, 0, 2, 2])
    assert_array_equal(X_noncanonical.data, [2, 0.5, 1.5, 1, 2])


def check_quantile_ranks_toy(name):
    """Check quantile ranks on toy data."""
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

    expected = [0.75, 0.75, 0.5, 1.0, 1.0, 1.0]
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
    """Check quantile ranks."""
    ForestRegressor = FOREST_REGRESSORS[name]

    # Check predicted ranks on (semi-)random data.
    x1 = np.random.choice(np.arange(0, 101), size=1000)
    e1_high = 10
    e1 = np.random.uniform(0, e1_high)

    X_train = x1.reshape(-1, 1)
    y_train = np.squeeze(x1 * 2 + e1)

    x2 = np.random.choice(np.arange(0, 101), size=10)

    X_test = x2.reshape(-1, 1)
    y_test = np.squeeze(X_test * 2 + e1)

    est = ForestRegressor(n_estimators=10, random_state=0)

    est.fit(X_train, y_train)
    y_ranks = est.quantile_ranks(X_test, y_test)

    assert y_ranks.shape == (X_test.shape[0],)
    assert np.all(y_ranks >= 0)
    assert np.all(y_ranks <= 1)

    # Check predicted ranks on multi-target data.
    y_train = np.stack([np.squeeze(x1), np.squeeze(x1 * 2)], axis=1)
    y_test = np.stack([np.squeeze(X_test - e1), np.squeeze(X_test * 2 + e1)], axis=1)

    est.fit(X_train, y_train)
    y_ranks = est.quantile_ranks(X_test, y_test)

    assert y_ranks.shape == (X_test.shape[0], 2)
    assert np.all(y_ranks >= 0)
    assert np.all(y_ranks <= 1)
    assert np.all(y_ranks[:, 0] < y_ranks[:, 1])

    # Check error if training and test number of targets are not equal.
    est.fit(X_train, y_train[:, 0])  # training target size = 1
    with pytest.raises(ValueError):
        est.quantile_ranks(X_test, y_test[:, :2])  # test target size = 2


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks(name):
    check_quantile_ranks(name)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks_keeps_fitted_features(name):
    """Check that quantile ranks do not reset the fitted feature attributes."""
    pd = pytest.importorskip("pandas")

    X, y = datasets.make_regression(n_samples=100, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, random_state=0).fit(X_df, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # X without feature names

        # Scoring a NumPy array keeps the feature names seen during fit.
        est.quantile_ranks(X, y)
        assert_array_equal(est.feature_names_in_, ["a", "b", "c"])

        # Scoring the wrong number of features raises without changing the model.
        with pytest.raises(ValueError):
            est.quantile_ranks(np.hstack([X, X]), y)
        assert est.n_features_in_ == 3

    est.predict(X_df)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks_monotonic_constraints(name):
    """Check that quantile ranks use the clipped leaf values that predict uses."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)

    ForestRegressor = FOREST_REGRESSORS[name]

    n_estimators = 25
    est = ForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        monotonic_cst=[1, 0],
        random_state=0,
    )
    est.fit(X, y)

    # With "nearest" interpolation, these quantiles are the clipped leaf values.
    quantiles = np.linspace(0, 1, n_estimators)
    leaf_vals = est.predict(X, quantiles=quantiles, interpolation="nearest")

    # Check ranks of scores between and equal to the clipped leaf values.
    for y_score in [y, leaf_vals[:, 5], leaf_vals[:, 20]]:
        for kind in ["rank", "weak", "strict", "mean"]:
            y_ranks = est.quantile_ranks(X, y_score, kind=kind)
            expected = [
                percentileofscore(v, s, kind=kind) / 100 for v, s in zip(leaf_vals, y_score)
            ]
            assert_allclose(y_ranks, expected)

    # Check ranks calculated for each tree and then averaged.
    for kind in ["rank", "weak", "strict", "mean"]:
        y_ranks = est.quantile_ranks(X[:20], y[:20], kind=kind, aggregate_leaves_first=False)
        expected = [
            np.mean([percentileofscore([v_i], s, kind=kind) / 100 for v_i in v])
            for v, s in zip(leaf_vals[:20], y[:20])
        ]
        assert_allclose(y_ranks, expected)

    # Check OOB ranks of the smallest and largest OOB predictions.
    est.set_params(n_estimators=50, bootstrap=True).fit(X, y)  # so all samples are OOB
    y_min = est.predict(X, quantiles=0, oob_score=True)
    y_max = est.predict(X, quantiles=1, oob_score=True)
    assert_array_equal(est.quantile_ranks(X, y_min, kind="strict", oob_score=True), 0)
    assert_array_equal(est.quantile_ranks(X, y_max, kind="weak", oob_score=True), 1)

    # Check errors for an invalid kind or number of outputs.
    with pytest.raises(ValueError, match="Invalid kind"):
        est.quantile_ranks(X, y, kind="invalid")
    with pytest.raises(ValueError, match="Number of target outputs"):
        est.quantile_ranks(X, np.stack([y, y], axis=1))


def check_proximity_counts(name):
    """Check proximity counts."""
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
    with pytest.raises(ValueError):
        est.proximity_counts(X, max_proximities=0)

    # Check error if `max_proximities` is a float.
    with pytest.raises(ValueError):
        est.proximity_counts(X, max_proximities=1.5)

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

    # Check proximity counts on regression dataset.
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)
    est = ForestRegressor(n_estimators=10, max_samples_leaf=None, bootstrap=True, random_state=0)
    est.fit(X, y)

    # Check that proximity counts match bootstrap counts.
    n_samples = len(X)
    proximities = est.proximity_counts(X)
    X_leaves = est.apply(X)
    args = (n_samples, n_samples)
    args = args if sklearn_version < parse_version("1.9") else args + (None,)
    bootstrap_indices = np.array(
        [_generate_sample_indices(e.random_state, *args) for e in est.estimators_]
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
    proximities_unsorted = est.proximity_counts(X, return_sorted=False)
    assert proximities != proximities_unsorted


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_proximity_counts(name):
    check_proximity_counts(name)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_proximity_counts_params(name):
    """Check the accepted types of `max_proximities` and the returned types."""
    X, y = datasets.make_regression(n_samples=100, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, max_samples_leaf=None, random_state=0)
    est.fit(X, y)

    proximities = est.proximity_counts(X[:5], max_proximities=3)
    assert est.proximity_counts(X[:5], max_proximities=np.int64(3)) == proximities

    for max_proximities in [0, 3.0, "3"]:
        with pytest.raises(ValueError):
            est.proximity_counts(X[:5], max_proximities=max_proximities)

    # Unsorted proximities are lists of tuples too.
    proximities = est.proximity_counts(X[:5], return_sorted=False)
    assert all(isinstance(p, list) for p in proximities)
    assert [sorted(p) for p in proximities] == [sorted(p) for p in est.proximity_counts(X[:5])]


def check_max_samples_leaf(name):
    """Check that the `max_samples_leaf` parameter correctly samples leaves."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    max_leaf_sizes = []
    for max_samples_leaf in [0.99 / len(X), 1, 3.0 / len(X), 5, 20, None]:
        est = ForestRegressor(
            n_estimators=10,
            min_samples_leaf=max_samples_leaf if max_samples_leaf is not None else len(X),
            max_samples_leaf=max_samples_leaf,
            random_state=0,
        )
        est.fit(X, y)

        max_leaf_size = 0
        y_train_leaves = est._get_y_train_leaves(X, y[:1])
        for i in range(len(y_train_leaves)):
            tree_lookup = y_train_leaves[i]
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
            with pytest.raises(ValueError):
                est.fit(X, y)
            est.max_samples_leaf = max_samples_leaf
            with pytest.raises(ValueError):
                est._get_y_train_leaves(X, y)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_max_samples_leaf(name):
    check_max_samples_leaf(name)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1.0, 1000])
def test_max_samples_leaf_above_leaf_sizes(name, max_samples_leaf):
    """Check that a limit above every leaf size keeps all samples, like None."""
    X, y = datasets.make_regression(n_samples=300, n_features=3, noise=1.0, random_state=0)
    params = {"n_estimators": 10, "min_samples_leaf": 10, "random_state": 0}

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(max_samples_leaf=max_samples_leaf, **params).fit(X, y)
    est_none = ForestRegressor(max_samples_leaf=None, **params).fit(X, y)

    # Leaves are padded to the largest leaf, not to the requested maximum.
    assert_array_equal(
        np.asarray(est.forest_.y_train_leaves), np.asarray(est_none.forest_.y_train_leaves)
    )

    # A float of 1.0 means all samples, so it must not use the 1-sample path.
    quantiles = [0.1, 0.5, 0.9]
    for weighted_quantile in [True, False]:
        assert_allclose(
            est.predict(X, quantiles=quantiles, weighted_quantile=weighted_quantile),
            est_none.predict(X, quantiles=quantiles, weighted_quantile=weighted_quantile),
        )


def check_missing_values(name):
    X = np.array(
        [
            [0, 1, 1, np.nan, np.nan, 0, 0, 0, 0, 0],
            [2, 2, 0, 1, 0, 0, 0, 0, 2, 2],
        ]
    ).T
    y = [2, 4, 5, 6, 7, 1, 1, 0, 1, 2]

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=1, max_samples_leaf=None, random_state=0)
    with pytest.raises(ValueError):
        est.fit(X, y)
    assert not hasattr(est, "estimators_")  # rejected before fitting the forest

    # Check that missing values are rejected when predicting too.
    est.fit(np.nan_to_num(X), y)
    for method in ["predict", "apply", "proximity_counts"]:
        with pytest.raises(ValueError):
            getattr(est, method)(X)
    with pytest.raises(ValueError):
        est.quantile_ranks(X, y)

    # Check that the estimator tags agree that missing values are not supported.
    assert not est._more_tags()["allow_nan"]  # the tags of scikit-learn < 1.6
    if sklearn_version >= parse_version("1.6"):
        from sklearn.utils import get_tags

        assert not get_tags(est).input_tags.allow_nan


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_missing_values(name):
    check_missing_values(name)


def check_oob_samples(name):
    """Check OOB sample generation."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

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
    """Check OOB sampling with duplicates."""
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


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_oob_samples_unsampled_masks(name):
    """Check that OOB samples come from masks of each tree's unsampled indices."""
    X, y = datasets.make_regression(n_samples=200, n_features=4, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, bootstrap=True, max_samples=0.7, random_state=0)
    for random_state in [0, 1]:  # check that refitting replaces the masks
        est.set_params(random_state=random_state).fit(X, y)
        masks = np.unpackbits(est._unsampled_masks, axis=1, count=len(X)).astype(bool)
        for mask, sample_indices in zip(masks, est.estimators_samples_):
            expected = np.setdiff1d(np.arange(len(X)), sample_indices)
            assert_array_equal(np.flatnonzero(mask), expected)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # Check that selected rows match the rows of all training samples.
        _, X_indices = est._oob_samples(X)
        indices = [5, -1, 17, 5]
        _, X_indices_selected = est._oob_samples(X[[5, 0, 17, 5]], indices=indices)
        assert_array_equal(X_indices_selected[[0, 2, 3]], X_indices[[5, 17, 5]])
        assert np.all(X_indices_selected[1] == 1)  # token index (-1) is always unsampled

        # Check that the masks are recomputed if missing, e.g., in older models.
        del est._unsampled_masks
        _, X_indices_recomputed = est._oob_samples(X)
        assert_array_equal(X_indices_recomputed, X_indices)

    # Check that no samples are out-of-bag without bootstrapping, with a warning.
    est = ForestRegressor(n_estimators=5, bootstrap=False, random_state=0).fit(X, y)
    assert est._unsampled_masks is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, X_indices = est._oob_samples(X)
    assert any("only exist if bootstrap=True" in str(w.message) for w in record)
    assert not X_indices.any()


def check_predict_oob(
    name,
    max_samples,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    """Check OOB predictions."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=20,
        max_samples_leaf=max_samples_leaf,
        max_samples=max_samples,
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
        assert math.isclose(est.oob_score_, y_pred_score)
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
    y_pred_precomputed_indices = est.predict(
        X,
        quantiles=quantiles,
        weighted_quantile=weighted_quantile,
        aggregate_leaves_first=aggregate_leaves_first,
        oob_score=True,
    )
    assert np.all(y_pred == y_pred_precomputed_indices)

    # Check single-row OOB scoring.
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
    y_pred_oob1 = est1.predict(X, quantiles=quantiles)
    y_pred_oob2 = est2.predict(X)
    assert_allclose(y_pred_oob1, y_pred_oob2)

    # Check error if OOB score without `indices` do not match training count.
    with pytest.raises(ValueError):
        est.predict(X[:1], oob_score=True)

    # Check error if OOB score with `indices` do not match samples count.
    with pytest.raises(ValueError):
        est.predict(
            X,
            oob_score=True,
            indices=-np.ones(len(X) - 1),
        )

    # Check warning if not enough estimators.
    with np.errstate(divide="ignore", invalid="ignore"):
        est = ForestRegressor(
            n_estimators=4,
            max_samples=max_samples,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        )
        with pytest.warns():
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
    with pytest.raises(ValueError):
        est.predict(
            X,
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert np.all(est._get_unsampled_indices(est.estimators_[0]) == np.array([]))

    # Check error if number of scoring and training samples are different.
    est = ForestRegressor(n_estimators=1, bootstrap=True)
    est.fit(X, y)
    with pytest.raises(ValueError):
        est.predict(
            X[:1],
            y[:1],
            weighted_quantile=weighted_quantile,
            aggregate_leaves_first=aggregate_leaves_first,
            oob_score=True,
        )


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples", [None, 0.5, 1.0, 100])
@pytest.mark.parametrize("max_samples_leaf", [None, 1])
@pytest.mark.parametrize("quantiles", [None, "mean", 0.5, [0.2, 0.5, 0.8]])
@pytest.mark.parametrize("weighted_quantile", [True, False])
@pytest.mark.parametrize("aggregate_leaves_first", [True, False])
def test_predict_oob(
    name,
    max_samples,
    max_samples_leaf,
    quantiles,
    weighted_quantile,
    aggregate_leaves_first,
):
    check_predict_oob(
        name,
        max_samples,
        max_samples_leaf,
        quantiles,
        weighted_quantile,
        aggregate_leaves_first,
    )


@pytest.mark.parametrize(
    "method", ["linear", "lower", "higher", "midpoint", "nearest", "hazen", "inverted_cdf"]
)
def test_nanquantile_rows(method):
    """Check that the row quantiles used for OOB predictions match np.nanquantile."""
    rng = np.random.default_rng(0)
    a = np.round(rng.normal(size=(300, 12)), 1)  # with ties
    a[rng.random(a.shape) < 0.5] = np.nan
    a[:5] = np.nan  # rows without values
    quantiles = [0.0, 0.1, 0.5, 0.75, 1.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows without values
        expected = np.nanquantile(a, quantiles, method=method, axis=1)
    assert_array_equal(_nanquantile_rows(a, quantiles, method), expected)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_predict_oob_single_sample_leaves(name):
    """Check OOB quantiles with max_samples_leaf=1 against each tree's OOB values."""
    X, y = datasets.make_regression(n_samples=200, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=10, bootstrap=True, random_state=0).fit(X, y)

    # Each tree's value for each sample, or NaN if the sample is in the tree's bootstrap.
    leaves = est.apply(X)
    y_train_leaves = np.asarray(est.forest_.y_train_leaves)
    y_train = np.asarray(est.forest_.y_train)[0]
    values = np.full(leaves.shape, np.nan)
    for tree, sample_indices in enumerate(est.estimators_samples_):
        oob = np.setdiff1d(np.arange(len(X)), sample_indices)
        values[oob, tree] = y_train[y_train_leaves[tree, leaves[oob, tree], 0, 0] - 1]

    quantiles = [0.1, 0.5, 0.9]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some samples have no OOB trees
        expected = np.nanquantile(values, quantiles, axis=1).T
        assert_allclose(est.predict(X, quantiles=quantiles, oob_score=True), expected)


def check_quantile_ranks_oob(name):
    """Check OOB quantile ranks."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

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
    with pytest.raises(ValueError):
        est.quantile_ranks(X, y, oob_score=True)

    # Check error if number of scoring and training samples are different.
    est = ForestRegressor(n_estimators=1, bootstrap=True)
    est.fit(X, y)
    with pytest.raises(ValueError):
        est.quantile_ranks(X[:1], y[:1], oob_score=True)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_quantile_ranks_oob(name):
    check_quantile_ranks_oob(name)


def check_proximity_counts_oob(name):
    """Check OOB proximity counts."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

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
    args = (n_samples, n_samples)
    args = args if sklearn_version < parse_version("1.9") else args + (None,)
    bootstrap_indices = np.array(
        [_generate_sample_indices(e.random_state, *args) for e in est.estimators_]
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
    with pytest.raises(ValueError):
        est.proximity_counts(X, oob_score=True)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_proximity_counts_oob(name):
    check_proximity_counts_oob(name)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_oob_methods_validate_once(name):
    """Check that OOB methods validate inputs once, not once per tree."""
    pd = pytest.importorskip("pandas")

    X, y = datasets.make_regression(n_samples=100, n_features=3, random_state=0)
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, bootstrap=True, random_state=0).fit(X_df, y)

    # A model fitted with feature names doesn't warn about them for each tree.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        y_pred = est.predict(X_df, oob_score=True)
        y_ranks = est.quantile_ranks(X_df, y, oob_score=True)
        est.proximity_counts(X_df, oob_score=True)
    assert not any("feature names" in str(w.message) for w in record)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # X without feature names, samples without OOB trees

        # Sparse inputs, which quantile_ranks validates as CSC, give the same results.
        X_sparse = sparse.csr_matrix(X)
        assert_allclose(est.predict(X_sparse, oob_score=True), y_pred)
        assert_allclose(est.quantile_ranks(X_sparse, y, oob_score=True), y_ranks)

        # Missing values are still rejected.
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError):
            est.quantile_ranks(X_nan, y, oob_score=True)


def check_monotonic_constraints(name, max_samples_leaf):
    """Check monotonic constraints."""
    ForestRegressor = FOREST_REGRESSORS[name]

    n_samples = 1000
    n_samples_train = 900

    # Build a regression task using 5 informative features.
    X, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        random_state=0,
    )
    train = np.arange(n_samples_train)
    test = np.arange(n_samples_train, n_samples)
    X_train = X[train]
    y_train = y[train]
    X_test = np.copy(X[test])
    y_test = np.copy(y[test])
    X_test_incr = np.copy(X_test)
    X_test_decr = np.copy(X_test)
    X_test_incr[:, 0] += 10
    X_test_decr[:, 1] += 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    est = ForestRegressor(
        max_samples_leaf=max_samples_leaf,
        monotonic_cst=monotonic_cst,
        max_leaf_nodes=n_samples_train,
        bootstrap=True,
    )

    for oob_score in [False, True]:
        if not oob_score:
            est.fit(X_train, y_train)
        else:
            est.fit(X_test, y_test)

        y = est.predict(X_test, oob_score=oob_score)

        score = est.score(X_test, y_test, quantiles=0.5)
        assert score > 0.75

        # Check the monotonic increase constraint.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_incr = est.predict(X_test_incr, oob_score=oob_score)
        assert np.all(y_incr >= y)

        # Check the monotonic decrease constraint.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_decr = est.predict(X_test_decr, oob_score=oob_score)
        assert np.all(y_decr <= y)

    # Check error if `max_samples_leaf` != 1.
    est = ForestRegressor(
        max_samples_leaf=None,
        monotonic_cst=monotonic_cst,
        max_leaf_nodes=n_samples_train,
        bootstrap=True,
    )
    with pytest.raises(ValueError):
        est.fit(X_train, y_train)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1])
def test_monotonic_constraints(name, max_samples_leaf):
    check_monotonic_constraints(name, max_samples_leaf)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_monotonic_bounds(name):
    """Check the leaf bounds for monotonicity against a node-by-node traversal."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] - X[:, 1] + rng.normal(size=300)
    monotonic_cst = [1, -1, 0]

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=3, monotonic_cst=monotonic_cst, random_state=0)
    est.fit(X, y)

    y_train = np.asarray(est.forest_.y_train)[0]
    y_train_leaves = np.asarray(est.forest_.y_train_leaves)
    y_bound_leaves = np.asarray(est.forest_.y_bound_leaves)
    for i, estimator in enumerate(est.estimators_):
        tree = estimator.tree_
        paths = estimator.decision_path(X).toarray().astype(bool)
        X_leaves = estimator.apply(X)

        def mean_target(node):
            """Return the mean of the stored targets of the leaves below `node`."""
            indices = y_train_leaves[i, np.unique(X_leaves[paths[:, node]])].ravel()
            return y_train[indices[indices != 0] - 1].mean()

        stack = [(0, -np.inf, np.inf)]  # (node, lower bound, upper bound)
        while stack:
            node, lower, upper = stack.pop()
            left, right = tree.children_left[node], tree.children_right[node]
            if left == right:  # leaf
                assert_allclose(y_bound_leaves[i, node], [lower, upper])
                continue
            cst = monotonic_cst[tree.feature[node]]
            mid = np.clip((mean_target(left) + mean_target(right)) / 2, lower, upper)
            if cst == 0:
                stack += [(left, lower, upper), (right, lower, upper)]
            elif cst == 1:
                stack += [(left, lower, mid), (right, mid, upper)]
            else:
                stack += [(left, mid, upper), (right, lower, mid)]


def check_serialization(name, sparse_pickle, monotonic_cst, multi_target):
    """Check model serialization/deserialization."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)

    if multi_target:
        y = np.vstack([y, y]).T

    if monotonic_cst and not multi_target:
        monotonic_cst = [1] * X.shape[1]
    else:
        monotonic_cst = None

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=10, monotonic_cst=monotonic_cst, random_state=0)
    est.fit(X, y, sparse_pickle=sparse_pickle)

    dumped = pickle.dumps(est)
    est_loaded = pickle.loads(dumped)

    assert check_is_fitted(est_loaded) is None
    assert np.all(est.predict(X) == est_loaded.predict(X))


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("sparse_pickle", [False, True])
@pytest.mark.parametrize("monotonic_cst", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
def test_serialization(name, sparse_pickle, monotonic_cst, multi_target):
    check_serialization(name, sparse_pickle, monotonic_cst, multi_target)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
@pytest.mark.parametrize("multi_target", [False, True])
def test_serialization_sparse_pickle(name, max_samples_leaf, multi_target):
    """Check that a model pickled as sparse matrices predicts as before."""
    X, y = datasets.make_regression(
        n_samples=100, n_features=2, n_targets=2 if multi_target else 1, random_state=0
    )

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=5, max_samples_leaf=max_samples_leaf, bootstrap=True, random_state=0
    )
    est.fit(X, y, sparse_pickle=True)
    est_loaded = pickle.loads(pickle.dumps(est))

    # The leaf arrays are restored in C order, which predictions rely on.
    assert np.asarray(est_loaded.forest_.y_train_leaves).flags.c_contiguous

    quantiles = [0.25, 0.5, 0.75]
    assert_allclose(est_loaded.predict(X, quantiles), est.predict(X, quantiles))
    assert_allclose(est_loaded.quantile_ranks(X, y), est.quantile_ranks(X, y))
    assert est_loaded.proximity_counts(X[:5]) == est.proximity_counts(X[:5])


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
@pytest.mark.parametrize("sparse_pickle", [False, True])
def test_serialization_memory_mapped(name, max_samples_leaf, sparse_pickle, tmp_path):
    """Check that a model loaded with read-only memory mapping works."""
    X, y = datasets.make_regression(n_samples=100, n_features=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(
        n_estimators=5, max_samples_leaf=max_samples_leaf, bootstrap=True, random_state=0
    )
    est.fit(X, y, sparse_pickle=sparse_pickle)

    joblib.dump(est, tmp_path / "model.joblib")
    est_loaded = joblib.load(tmp_path / "model.joblib", mmap_mode="r")

    quantiles = [0.25, 0.5, 0.75]
    assert_allclose(est_loaded.predict(X, quantiles), est.predict(X, quantiles))
    assert_allclose(est_loaded.quantile_ranks(X, y), est.quantile_ranks(X, y))
    assert est_loaded.proximity_counts(X[:5]) == est.proximity_counts(X[:5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some samples have no OOB predictions
        assert_allclose(
            est_loaded.predict(X, quantiles, oob_score=True),
            est.predict(X, quantiles, oob_score=True),
        )


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_y_train_is_array(name):
    """Check that the training targets are stored as an array, read without copying."""
    X, y = datasets.make_regression(n_samples=100, n_features=2, n_targets=2, random_state=0)

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, random_state=0).fit(X, y)

    y_train = np.asarray(est.forest_.y_train)
    assert y_train.dtype == np.float64
    assert_array_equal(y_train, np.sort(y, axis=0).T)  # sorted, one row per output
    assert np.shares_memory(y_train, np.asarray(est.forest_.y_train))  # not a copy


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_sample_weight_stored_on_fit(name):
    """Check that sample_weight is stored during fit."""
    X, y = datasets.make_regression(n_samples=100, n_features=4, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, random_state=0)
    est.fit(X, y, sample_weight=sample_weight)
    assert est.sample_weight_ is not None
    assert est.sample_weight_.shape[0] == len(y)

    est_no_weight = ForestRegressor(n_estimators=5, random_state=0)
    est_no_weight.fit(X, y)
    assert est_no_weight.sample_weight_ is None


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("bootstrap", [False, True])
def test_sample_weight_validation(name, bootstrap):
    """Check that sample weights are accepted in any form scikit-learn accepts."""
    X, y = datasets.make_regression(n_samples=100, n_features=4, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))
    params = {"n_estimators": 5, "bootstrap": bootstrap, "random_state": 0}
    if bootstrap:
        params["max_samples"] = 0.5  # draws relative to the weights on sklearn 1.9+
    quantiles = [0.25, 0.5, 0.75]

    ForestRegressor = FOREST_REGRESSORS[name]

    # A list of weights gives the same model as an array of weights.
    est = ForestRegressor(**params).fit(X, y, sample_weight=sample_weight)
    est_list = ForestRegressor(**params).fit(X, y, sample_weight=sample_weight.tolist())
    assert isinstance(est_list.sample_weight_, np.ndarray)
    assert_allclose(est_list.predict(X, quantiles), est.predict(X, quantiles))
    if bootstrap:
        kwargs = {"quantiles": quantiles, "oob_score": True}
        assert_allclose(est_list.predict(X, **kwargs), est.predict(X, **kwargs))

    # A scalar weight is applied to every sample.
    est = ForestRegressor(**params).fit(X, y, sample_weight=np.full(len(y), 2.0))
    est_scalar = ForestRegressor(**params).fit(X, y, sample_weight=2.0)
    assert_array_equal(est_scalar.sample_weight_, np.full(len(y), 2.0))
    assert_allclose(est_scalar.predict(X, quantiles), est.predict(X, quantiles))

    # Weights that only support conversion to an array work too.
    est = ForestRegressor(n_estimators=5, bootstrap=bootstrap, random_state=0)
    check_sample_weights_not_an_array(name, est)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_sample_weight_predict(name):
    """Check that predictions work with sample_weight on sklearn 1.9+."""
    X, y = datasets.make_regression(n_samples=200, n_features=4, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=10, bootstrap=True, random_state=0)
    est.fit(X, y, sample_weight=sample_weight)

    y_pred = est.predict(X, quantiles=[0.25, 0.5, 0.75])
    assert y_pred.shape == (len(X), 3)
    assert not np.isnan(y_pred).any()

    # Predictions with sample_weight should differ from uniform weights.
    est_uniform = ForestRegressor(n_estimators=10, bootstrap=True, random_state=0)
    est_uniform.fit(X, y)
    y_pred_uniform = est_uniform.predict(X, quantiles=[0.25, 0.5, 0.75])
    assert not np.allclose(y_pred, y_pred_uniform)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_sample_weight_oob(name):
    """Check that OOB predictions work with sample_weight on sklearn 1.9+."""
    X, y = datasets.make_regression(n_samples=500, n_features=8, noise=0.1, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=20, bootstrap=True, oob_score=True, random_state=0)
    est.fit(X, y, sample_weight=sample_weight)

    y_pred = est.predict(X, quantiles=[0.2, 0.5, 0.8], oob_score=True)
    assert y_pred.shape == (len(X), 3)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_sample_weight_unsampled_indices(name):
    """Check that _get_unsampled_indices uses sample_weight on sklearn 1.9+."""
    X, y = datasets.make_regression(n_samples=200, n_features=4, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, bootstrap=True, random_state=0)
    est.fit(X, y, sample_weight=sample_weight)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unsampled = est._get_unsampled_indices(est.estimators_[0])

    assert len(unsampled) > 0
    assert all(idx < len(y) for idx in unsampled)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_sample_weight_proximity_counts(name):
    """Check that proximity_counts works with sample_weight on sklearn 1.9+."""
    X, y = datasets.make_regression(n_samples=100, n_features=4, random_state=0)
    sample_weight = np.random.default_rng(0).uniform(0.1, 2.0, size=len(y))

    ForestRegressor = FOREST_REGRESSORS[name]

    est = ForestRegressor(n_estimators=5, bootstrap=True, random_state=0)
    est.fit(X, y, sample_weight=sample_weight)

    proximities = est.proximity_counts(X)
    assert len(proximities) == len(X)


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
def test_sample_weight_zero_weight_targets(name, bootstrap, multi_target):
    """Exclude zero-weight samples using their original training indices."""
    X = np.zeros((4, 1))
    y = np.array([10, 40, 20, 30])
    if multi_target:
        y = np.column_stack((y, -y))
    sample_weight = np.array([0, 1, 1, 1])
    quantiles = [0.0, 0.5, 1.0]
    est = FOREST_REGRESSORS[name](
        n_estimators=1, bootstrap=bootstrap, max_samples_leaf=None, random_state=0
    )
    est.fit(X, y, sample_weight=sample_weight)

    indices = est.estimators_samples_[0]
    indices = indices[sample_weight[indices] > 0]
    expected = np.quantile(y[indices], quantiles, axis=0).T
    assert_allclose(est.predict(X[:1], quantiles=quantiles), expected[np.newaxis])


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("max_samples_leaf", [1, None])
def test_sample_weight_single_sample(name, bootstrap, multi_target, max_samples_leaf):
    """Keep weight indexing valid for one training sample and any output count."""
    X = [[0]]
    y = [[2, -2]] if multi_target else [2]
    est = FOREST_REGRESSORS[name](
        n_estimators=1,
        bootstrap=bootstrap,
        max_samples_leaf=max_samples_leaf,
        random_state=0,
    )
    est.fit(X, y, sample_weight=[1])
    assert_allclose(est.predict(X), y)


@pytest.mark.skipif(
    sklearn_version < parse_version("1.9"),
    reason="sample_weight in bootstrap functions requires scikit-learn >= 1.9",
)
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("multi_target", [False, True])
def test_sample_weight_bootstrap_more_than_n_samples(name, multi_target):
    """Collect weighted bootstrap rows when their count exceeds the training size."""
    X = np.zeros((4, 1))
    y = np.array([10, 40, 20, 30])
    if multi_target:
        y = np.column_stack((y, -y))
    sample_weight = np.full(len(X), 2.0)
    quantiles = [0.0, 0.5, 1.0]
    est = FOREST_REGRESSORS[name](
        n_estimators=1,
        bootstrap=True,
        max_samples=1.0,
        max_samples_leaf=None,
        random_state=0,
    )
    est.fit(X, y, sample_weight=sample_weight)

    indices = est.estimators_samples_[0]
    assert len(indices) == 2 * len(X)
    expected = np.quantile(y[indices], quantiles, axis=0).T
    assert_allclose(est.predict(X[:1], quantiles=quantiles), expected[np.newaxis])


def test_calc_quantile():
    """Check quantile calculations."""
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
    with pytest.raises(TypeError):
        calc_quantile([1, 2], 0.5)
    with pytest.raises(TypeError):
        calc_quantile([1, 2], [0.5], interpolation=None)


def test_calc_weighted_quantile():
    """Check weighted quantile calculations."""
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
    with pytest.raises(TypeError):
        calc_weighted_quantile([1, 2], [1, 1], 0.5)
    with pytest.raises(TypeError):
        calc_weighted_quantile([1, 2], [1, 1], [0.5], interpolation=None)


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_calc_quantile_matches_numpy(interpolation):
    """Check that quantiles match NumPy, including weights given as repeats."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 15))
        inputs = np.sort(rng.integers(0, 5, size=n).astype(float))  # many ties
        weights = rng.integers(0, 4, size=n).astype(float)  # some zero weights
        weights[rng.integers(n)] += 1
        repeated = np.repeat(inputs, weights.astype(int))

        # Include quantiles that land exactly on a rank of the repeated values.
        quantiles = list(rng.random(3)) + [0.0, 1.0]
        if len(repeated) > 1:
            quantiles += list(rng.integers(0, len(repeated), size=3) / (len(repeated) - 1))

        expected = np.quantile(inputs, quantiles, method=interpolation)
        actual = calc_quantile(inputs, quantiles, interpolation.encode(), issorted=True)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        expected = np.quantile(repeated, quantiles, method=interpolation)
        actual = calc_weighted_quantile(
            inputs, weights, quantiles, interpolation.encode(), issorted=True
        )
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("C, method", [(0, "weibull"), (0.5, "hazen"), (1, "linear")])
@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_calc_quantile_rank_constant(C, method, interpolation):
    """Check quantiles for each constant C in the rank, including weighted ones."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 15))
        inputs = np.sort(rng.integers(0, 5, size=n).astype(float))  # many ties
        weights = rng.integers(0, 4, size=n).astype(float)  # some zero weights
        weights[rng.integers(n)] += 1
        repeated = np.repeat(inputs, weights.astype(int))
        n_repeated = len(repeated)

        # Include the ends, which C < 1 can index past, and exact ranks.
        quantiles = list(rng.random(3)) + [0.0, 1.0]
        if n_repeated + 1 - 2 * C > 0:
            ranks = rng.integers(0, n_repeated, size=3)
            quantiles += list((ranks + 1 - C) / (n_repeated + 1 - 2 * C))

        kwargs = {"issorted": True, "C": C}
        expected = calc_quantile(repeated, quantiles, interpolation.encode(), **kwargs)
        if interpolation == "linear":
            # Check against NumPy's method for this constant.
            assert_allclose(
                expected, np.quantile(repeated, quantiles, method=method), rtol=1e-12, atol=1e-12
            )

        # Check that integer weights match the quantiles of the repeated values.
        actual = calc_weighted_quantile(
            inputs, weights, quantiles, interpolation.encode(), **kwargs
        )
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_calc_weighted_quantile_ignores_zero_weights(interpolation):
    """Check that zero-weight inputs never affect the weighted quantile."""
    inputs = [-0.7, -0.1, 0.2, 1.1, 1.5]
    weights = [0, 3, 3, 0, 0]

    # The lowest input has zero weight, so q=0 must return the next input.
    actual = calc_weighted_quantile(inputs, weights, [0.0, 1.0], interpolation.encode())
    assert_allclose(actual, [-0.1, 0.2])

    # Fractional weights: q=1 must reach the last input with positive weight.
    actual = calc_weighted_quantile([0.0, 1.0], [1.0, 0.05], [1.0], interpolation.encode())
    assert_allclose(actual, [1.0])


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_predict_weighted_matches_unweighted(name, interpolation):
    """Check that weighted and unweighted quantiles agree with count weights."""
    X, y = datasets.make_regression(n_samples=400, n_features=4, noise=10, random_state=2)
    est = FOREST_REGRESSORS[name](
        n_estimators=5, max_samples_leaf=None, min_samples_leaf=20, random_state=2
    )
    est.fit(X, y)
    X_test = np.random.default_rng(2).normal(size=(200, 4))

    # With leaf weights off, the weights are counts, so both must agree.
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    kwargs = {"quantiles": quantiles, "interpolation": interpolation}
    assert_allclose(
        est.predict(X_test, weighted_quantile=True, **kwargs),
        est.predict(X_test, weighted_quantile=False, **kwargs),
    )


def test_calc_quantile_rank():
    """Check quantile rank calculations."""
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
    scores = float(1)

    # Check that -1 is returned for empty list.
    actual = calc_quantile_rank(inputs, scores)
    expected = [-1]
    assert_array_equal(actual, expected)

    inputs = [2, 4, 3]
    scores = float(3)

    # Check that sorting is correctly applied.
    actual1 = calc_quantile_rank(inputs, scores, issorted=True)
    actual2 = calc_quantile_rank(inputs, scores, issorted=False)
    assert actual1 != actual2

    # Check error if invalid parameters.
    with pytest.raises(TypeError):
        calc_quantile_rank([1, 2], [1])
    with pytest.raises(TypeError):
        calc_quantile_rank([1, 2], float(1), kind=None)
