"""
================================
Multi-target Quantile Regression
================================

An example on a toy dataset that demonstrates fitting a single quantile
regressor for multiple target variables. For each target, multiple quantiles
can be estimated.

"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)


n_samples = 10000
bounds = [-0, 25]
funcs = [
    lambda x: np.sin(x) + np.sqrt(x),
    lambda x: np.cos(x),
    lambda x: np.sin(x) - np.sqrt(x),
]


def make_func_Xy(funcs, bounds, n_samples):
    x = np.linspace(bounds[0], bounds[1], n_samples)
    y = np.empty((len(x), 3))
    y[:, 0] = funcs[0](x) + np.random.normal(scale=0.01 * np.abs(x))
    y[:, 1] = funcs[1](x) + np.random.normal(scale=0.01 * np.abs(x))
    y[:, 2] = funcs[2](x) + np.random.normal(scale=0.01 * np.abs(x))
    return x, y


X, y = make_func_Xy(funcs, bounds, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

qrf = RandomForestQuantileRegressor(random_state=0)
qrf.fit(X_train.reshape(-1, 1), y_train)

y_pred = qrf.predict(X.reshape(-1, 1), quantiles=[0.025, 0.5, 0.975])


def plot_multioutputs(colors, funcs, X, y):
    for i in range(y.shape[-1]):
        plt.fill_between(X, y_pred[:, 0, i], y_pred[:, 2, i], color=colors[i], label=f"Target {i}")
        plt.plot(X, funcs[i](X), c="black")
    plt.xlim(bounds)
    plt.ylim([-8, 8])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(loc="upper left")
    plt.title("Multi-target Prediction Intervals")
    plt.show()


colors = ["#f2a619", "#006aff", "#001751"]
plot_multioutputs(colors, funcs, X, y)