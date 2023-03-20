"""
========================================================================
Quantile regression forest predictions compared to ground truth function
========================================================================

An example that demonstrates the use of a quantile regression forest to
predict a conditional median and prediction intervals. The example compares
the predictions to a ground truth function used to generate noisy samples.

"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor


def make_toy_dataset(n_samples, seed=0):
    rng = np.random.RandomState(seed)

    x = rng.uniform(0, 10, size=n_samples)
    f = x * np.sin(x)

    sigma = 0.25 + x / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f + noise

    return np.atleast_2d(x).T, y


n_samples = 1000
X, y = make_toy_dataset(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

xx = np.atleast_2d(np.linspace(0, 10, n_samples)).T

qrf = RandomForestQuantileRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
qrf.fit(X_train, y_train)

y_pred = qrf.predict(xx, quantiles=[0.025, 0.5, 0.975])

y_pred_low = y_pred[:, 0]
y_pred_med = y_pred[:, 1]
y_pred_upp = y_pred[:, 2]

plt.plot(X_test, y_test, ".", c="#f2a619", label="Test Observations", ms=5)
plt.plot(xx, (xx * np.sin(xx)), c="black", label="$f(x) = x\,\sin(x)$", lw=2)
plt.plot(xx, y_pred_med, c="#006aff", label="Predicted Median", lw=3, ms=5)
plt.fill_between(
    xx.ravel(),
    y_pred_low,
    y_pred_upp,
    color="#e0f2ff",
    label="Predicted 95% Interval",
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend(loc="upper left")
plt.show()
