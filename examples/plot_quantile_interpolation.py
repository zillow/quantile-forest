"""
========================================================
Predicting with different quantile interpolation methods
========================================================

An example comparison of interpolation methods that can be applied during
prediction when the desired quantile lies between two data points.

"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from quantile_forest import RandomForestQuantileRegressor


# Create toy dataset.
X = np.array([[-2, -2], [-2, -2], [-1, -1], [-1, -1], [1, 1], [1, 2]])
y = np.array([-1, -1, 0, 1, 1, 2])

est = RandomForestQuantileRegressor(
    n_estimators=1,
    bootstrap=False,
    random_state=0,
)
est.fit(X, y)

interpolations = ["linear", "lower", "higher", "midpoint", "nearest"]
colors = ["#eeca3b", "#4c78a8", "#72b7b2", "#54a24b", "#f58518"]

y_medians = []
y_errs = []

for interpolation in interpolations:
    y_pred = est.predict(
        X,
        quantiles=[0.025, 0.5, 0.975],
        interpolation=interpolation,
    )
    y_medians.append(y_pred[:, 1])
    y_errs.append(np.concatenate(([y_pred[:, 2]], [y_pred[:, 0]]), axis=0))

sc = plt.scatter(np.arange(len(y)) - .35, y, color="k", zorder=10)
ebs = []
for i, (median, y_err) in enumerate(zip(y_medians, y_errs)):
    ebs.append(
        plt.errorbar(
            np.arange(len(y)) + (.15 * (i + 1)) - .35,
            median,
            yerr=y_err,
            color=colors[i],
            ecolor=colors[i],
            fmt="o",
        )
    )
plt.xlim([-.75, len(y) - .25])
plt.xticks(np.arange(len(y)), X.tolist())
plt.xlabel("Samples (Feature Values)")
plt.ylabel("Actual and Predicted Values")
plt.legend([sc] + ebs, ["actual"] + interpolations, loc=2)
plt.show()
