"""Testing for the use of the estimators from threads, as with free-threaded Python."""

import os
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn import datasets
from sklearn.utils._testing import assert_array_equal

from quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor

FOREST_REGRESSORS = {
    "ExtraTreesQuantileRegressor": ExtraTreesQuantileRegressor,
    "RandomForestQuantileRegressor": RandomForestQuantileRegressor,
}


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="Needs free-threaded Python."
)
def test_import_keeps_gil_disabled():
    """Check that importing the package does not enable the GIL."""
    code = "import sys; import quantile_forest; print(sys._is_gil_enabled())"
    env = {name: value for name, value in os.environ.items() if name != "PYTHON_GIL"}
    process = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert process.stdout.strip() == "False", process.stderr


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
@pytest.mark.parametrize("max_samples_leaf", [1, None])
def test_methods_in_threads(name, max_samples_leaf):
    """Check that methods return the same when threads call them at once."""
    X, y = datasets.make_regression(n_samples=200, n_features=4, random_state=0)
    est = FOREST_REGRESSORS[name](
        n_estimators=30, max_samples_leaf=max_samples_leaf, bootstrap=True, random_state=0
    )
    est.fit(X, y)

    methods = [
        lambda: est.predict(X, quantiles=[0.1, 0.5, 0.9]),
        lambda: est.predict(X, quantiles="mean"),
        lambda: est.predict(X, quantiles=[0.5], weighted_leaves=True),
        lambda: est.predict(X, quantiles=[0.5], oob_score=True),
        lambda: est.quantile_ranks(X, y),
        lambda: est.proximity_counts(X),
    ]
    expected = [method() for method in methods]

    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def call_methods(_):
        barrier.wait()  # so that the threads call the methods at once
        return [method() for method in methods]

    with ThreadPoolExecutor(n_threads) as executor:
        for actual in executor.map(call_methods, range(n_threads)):
            for actual_result, expected_result in zip(actual, expected):
                if isinstance(expected_result, np.ndarray):
                    assert_array_equal(actual_result, expected_result)
                else:
                    assert actual_result == expected_result
