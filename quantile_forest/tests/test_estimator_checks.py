"""Testing that the estimators pass the estimator checks of scikit-learn."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from quantile_forest import ExtraTreesQuantileRegressor, RandomForestQuantileRegressor


def _expected_failed_checks(estimator):
    """Return the checks that an estimator is expected to fail, with the reason."""
    if not estimator.bootstrap:
        return {}
    # A forest draws its bootstrap samples at random, so these checks only pass by chance,
    # as for the forests of scikit-learn, which expects them to fail the checks too.
    reason = "sample_weight is not equivalent to removing/repeating samples."
    return {
        "check_sample_weight_equivalence_on_dense_data": reason,
        "check_sample_weight_equivalence_on_sparse_data": reason,
    }


@parametrize_with_checks(
    [
        Forest(n_estimators=10, max_samples_leaf=max_samples_leaf)
        for Forest in (ExtraTreesQuantileRegressor, RandomForestQuantileRegressor)
        for max_samples_leaf in (1, None)
    ],
    expected_failed_checks=_expected_failed_checks,
    xfail_strict=False,
)
def test_sklearn_compatible_estimator(estimator, check):
    """Check that the estimators adhere to the conventions of scikit-learn."""
    check(estimator)
