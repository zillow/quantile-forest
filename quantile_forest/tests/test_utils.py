"""Testing for the utils module (quantile_forest._utils)."""

import numpy as np
from sklearn.utils._testing import assert_array_equal

from quantile_forest._utils import group_indices_by_value


def test_group_indices_by_value():
    """Check grouping indices by value."""
    inputs = np.array([1, 3, 2, 2, 5, 4, 5, 5], dtype=np.int64)

    actual_indices, actual_values = group_indices_by_value(inputs)
    expected_indices = np.array([1, 2, 3, 4, 5])
    expected_values = [
        np.array([0]),
        np.array([2, 3]),
        np.array([1]),
        np.array([5]),
        np.array([4, 6, 7]),
    ]
    assert_array_equal(actual_indices, expected_indices)
    for actual, expected in zip(actual_values, expected_values):
        assert_array_equal(actual, expected)


def test_group_indices_by_value_stable():
    """Check that each group keeps its indices in their original order."""
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 100, size=10_000).astype(np.int64)  # many ties

    values, groups = group_indices_by_value(inputs)
    assert_array_equal(values, np.unique(inputs))
    for value, group in zip(values, groups):
        assert_array_equal(group, np.flatnonzero(inputs == value))
