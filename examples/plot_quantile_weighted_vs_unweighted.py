"""
=================================================
Predicting with weighted and unweighted quantiles
=================================================

An example comparison of the prediction runtime when using a quantile
regression forest with weighted and unweighted quantiles to compute the
predicted output values. A standard random forest regressor is included for
comparison.

"""
print(__doc__)

import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor


@contextmanager
def timing():
    t0 = time.time()
    yield lambda: (t1 - t0)
    t1 = time.time()


# Create synthetic regression dataset.
X, y = datasets.make_regression(n_samples=1000, n_features=4, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

estimator_sizes = [1, 5, 10, 25, 50, 100]

n_repeats = 10
n_sizes = len(estimator_sizes)

timings = []

timings = np.empty((n_sizes, n_repeats, 3))
for i, n_estimators in enumerate(estimator_sizes):
    for j in range(n_repeats):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=0,
        )
        qrf = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            max_samples_leaf=None,
            random_state=0,
        )

        rf.fit(X_train, y_train)
        qrf.fit(X_train, y_train)

        with timing() as rf_time:
            _ = rf.predict(X_test)
        with timing() as qrf_weighted_time:
            _ = qrf.predict(X_test, quantiles=0.5, weighted_quantile=True)
        with timing() as qrf_unweighted_time:
            _ = qrf.predict(X_test, quantiles=0.5, weighted_quantile=False)
        timings[i, j, :] = [rf_time(), qrf_weighted_time(), qrf_unweighted_time()]
        timings[i, j, :] *= 1000

rf_time_avg, qrf_weighted_time_avg, qrf_unweighted_time_avg = list(zip(*np.mean(timings, axis=1)))
rf_time_std, qrf_weighted_time_std, qrf_unweighted_time_std = list(zip(*np.std(timings, axis=1)))

plt.plot(estimator_sizes, rf_time_avg, c="#f2a619")
plt.plot(estimator_sizes, qrf_weighted_time_avg, c="#006aff")
plt.plot(estimator_sizes, qrf_unweighted_time_avg, c="#001751")
plt.fill_between(
    estimator_sizes,
    np.array(rf_time_avg) - (np.array(rf_time_std) / 2),
    np.array(rf_time_avg) + (np.array(rf_time_std) / 2),
    alpha=0.1,
    color="#f2a619",
)
plt.fill_between(
    estimator_sizes,
    np.array(qrf_weighted_time_avg) - (np.array(qrf_weighted_time_std) * 1.96),
    np.array(qrf_weighted_time_avg) + (np.array(qrf_weighted_time_std) * 1.96),
    alpha=0.1,
    color="#006aff",
)
plt.fill_between(
    estimator_sizes,
    np.array(qrf_unweighted_time_avg) - (np.array(qrf_unweighted_time_std) * 1.96),
    np.array(qrf_unweighted_time_avg) + (np.array(qrf_unweighted_time_std) * 1.96),
    alpha=0.1,
    color="#001751",
)

plt.xlim([estimator_sizes[0], estimator_sizes[-1]])
plt.xlabel("Number of Estimators")
plt.ylabel("Prediction Runtime (seconds)")
plt.legend(["RF", "QRF Weighted Quantile", "QRF Unweighted Quantile"])
plt.show()
