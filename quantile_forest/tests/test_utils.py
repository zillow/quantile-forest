"""Testing for the utils module (quantile_forest._utils)."""

import numpy as np
from sklearn.utils._testing import assert_array_equal

from quantile_forest._utils import (
    generate_unsampled_indices,
    group_indices_by_value,
    map_indices_to_leaves,
)


def test_generate_unsampled_indices():
    """Check unsampled indices generation."""
    max_index = 20
    duplicates = [[1, 4], [19, 10], [2, 3, 5], [6, 13]]

    def _generate_unsampled_indices(sample_indices, n_total_samples):
        return generate_unsampled_indices(
            np.array(sample_indices, dtype=np.int64),
            n_total_samples=n_total_samples,
            duplicates=duplicates,
        )

    # If all indices are sampled, there are no unsampled indices.
    indices = [idx for idx in range(max_index)]
    expected = np.array([], dtype=np.int64)
    assert_array_equal(_generate_unsampled_indices(indices, max_index), expected)

    # Index 7 has no duplicates, and thus should be the only unsampled index.
    indices = [7 for _ in range(max_index)]
    expected = np.array([idx for idx in range(max_index) if idx != 7])
    assert_array_equal(_generate_unsampled_indices(indices, max_index), expected)

    # Check sample indices [0, 1, 2] with duplicates set(1, 4) + set(2, 3, 5),
    # which excludes [0, 1, 2, 3, 4, 5] (i.e., range(6)) from unsampled.
    indices = [idx % 3 for idx in range(max_index)]
    expected = [x for x in range(max_index) if x not in range(6)]
    assert_array_equal(_generate_unsampled_indices(indices, max_index), expected)


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


def test_map_indices_to_leaves():
    """Check mapping of indices to leaf nodes."""
    y_train_leaves = np.zeros((3, 1, 3), dtype=np.int64)
    bootstrap_indices = np.array([[1], [2], [3], [4], [5]], dtype=np.int64)
    leaf_indices = np.array([1, 2])
    leaf_values_list = [np.array([0, 1, 2]), np.array([3, 4])]

    actual = map_indices_to_leaves(
        y_train_leaves,
        bootstrap_indices,
        leaf_indices,
        leaf_values_list,
    )
    expected = np.array([[[0, 0, 0]], [[1, 2, 3]], [[4, 5, 0]]])
    assert_array_equal(actual, expected)
