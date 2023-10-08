"""
===============================================
Quantile regression forest prediction intervals
===============================================

An example of how to use a quantile regression forest to plot prediction
intervals on the California Housing dataset.

"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

rng = check_random_state(0)

# Load the California Housing Prices dataset.
california = datasets.fetch_california_housing()
n_samples = min(california.target.size, 1000)
perm = rng.permutation(n_samples)
X = california.data[perm]
y = california.target[perm]

qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=0)

kf = KFold(n_splits=5)
kf.get_n_splits(X)

y_true = []
y_pred = []
y_pred_low = []
y_pred_upp = []

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = (
        X[train_index],
        X[test_index],
        y[train_index],
        y[test_index],
    )

    qrf.set_params(max_features=X_train.shape[1] // 3)
    qrf.fit(X_train, y_train)

    # Get predictions at 95% prediction intervals and median.
    y_pred_i = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])

    y_true = np.concatenate((y_true, y_test))
    y_pred = np.concatenate((y_pred, y_pred_i[:, 1]))
    y_pred_low = np.concatenate((y_pred_low, y_pred_i[:, 0]))
    y_pred_upp = np.concatenate((y_pred_upp, y_pred_i[:, 2]))


def plot_calibration_and_intervals(y_true, y_pred, y_pred_low, y_pred_upp):
    def plot_calibration(ax, y_true, y_pred_low, y_pred_upp, price_formatter):
        y_min = min(np.minimum(y_true, y_pred))
        y_max = max(np.maximum(y_true, y_pred))

        for low, mid, upp in zip(y_pred_low, y_pred, y_pred_upp):
            ax.plot([mid, mid], [low, upp], lw=4, c="#e0f2ff")

        ax.plot(y_pred, y_true, c="#f2a619", lw=0, marker=".", ms=5)
        ax.plot(y_pred, y_pred_low, alpha=0.4, c="#006aff", lw=0, marker="_", ms=4)
        ax.plot(y_pred, y_pred_upp, alpha=0.4, c="#006aff", lw=0, marker="_", ms=4)
        ax.plot([y_min, y_max], [y_min, y_max], ls="--", lw=1, c="grey")
        ax.grid(axis="x", color="0.95")
        ax.grid(axis="y", color="0.95")
        ax.xaxis.set_major_formatter(price_formatter)
        ax.yaxis.set_major_formatter(price_formatter)
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Fitted Values (Conditional Median)")
        ax.set_ylabel("Observed Values")

    def plot_intervals(ax, y_true, y_pred_low, y_pred_upp, price_formatter):
        # Center data, with the mean of the prediction interval at 0.
        mean = (y_pred_low + y_pred_upp) / 2
        y_true -= mean
        y_pred_low -= mean
        y_pred_upp -= mean

        ax.plot(y_true, c="#f2a619", lw=0, marker=".", ms=5)
        ax.fill_between(
            np.arange(len(y_pred_upp)),
            y_pred_low,
            y_pred_upp,
            alpha=0.8,
            color="#e0f2ff",
        )
        ax.plot(np.arange(n_samples), y_pred_low, alpha=0.8, c="#006aff", lw=2)
        ax.plot(np.arange(n_samples), y_pred_upp, alpha=0.8, c="#006aff", lw=2)
        ax.grid(axis="x", color="0.95")
        ax.grid(axis="y", color="0.95")
        ax.yaxis.set_major_formatter(price_formatter)
        ax.set_xlim([0, n_samples])
        ax.set_xlabel("Ordered Samples")
        ax.set_ylabel("Observed Values and Prediction Intervals")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    usd_formatter = FuncFormatter(lambda x, p: f"${format(int(x) * 100, ',')}k")

    y_pred_interval = y_pred_upp - y_pred_low
    sort_idx = np.argsort(y_pred)
    y_true = y_true[sort_idx]
    y_pred = y_pred[sort_idx]
    y_pred_low = y_pred_low[sort_idx]
    y_pred_upp = y_pred_upp[sort_idx]

    plot_calibration(ax1, y_true, y_pred_low, y_pred_upp, usd_formatter)

    y_pred_interval = y_pred_upp - y_pred_low
    sort_idx = np.argsort(y_pred_interval)
    y_true = y_true[sort_idx]
    y_pred_low = y_pred_low[sort_idx]
    y_pred_upp = y_pred_upp[sort_idx]

    plot_intervals(ax2, y_true, y_pred_low, y_pred_upp, usd_formatter)

    plt.subplots_adjust(top=0.15)
    fig.tight_layout(pad=3)

    plt.show()


plot_calibration_and_intervals(y_true, y_pred, y_pred_low, y_pred_upp)
