"""
=================================================
Predicting with weighted and unweighted quantiles
=================================================

An example comparison of the prediction runtime when using a quantile
regression forest with weighted and unweighted quantiles to compute the
predicted output values.

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
X, y = datasets.make_regression(n_samples=500, n_features=4, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

estimator_sizes = [1, 5, 10, 25, 50, 100, 250, 500]
n_repeats = 3

timings = []

for n_estimators in estimator_sizes:
    timings_i = []
    for i in range(n_repeats):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=0,
        )
        qrf = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
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
        timings_i.append(
            (
                rf_time(),
                qrf_weighted_time(),
                qrf_unweighted_time(),
            )
        )

    timings.append([np.min(x) * 1000. for x in list(zip(*timings_i))])

rf_time, qrf_weighted_time, qrf_unweighted_time = list(zip(*timings))

line_chart1 = plt.plot(estimator_sizes, rf_time)
line_chart2 = plt.plot(estimator_sizes, qrf_weighted_time)
line_chart3 = plt.plot(estimator_sizes, qrf_unweighted_time)
plt.xlim([estimator_sizes[0], estimator_sizes[-1]])
plt.xlabel("Number of Estimators")
plt.ylabel("Prediction Runtime (ms)")
plt.legend(["RF", "QRF Weighted Quantile", "QRF Unweighted Quantile"])
plt.show()
