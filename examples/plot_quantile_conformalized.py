"""
=================================================================
Quantile regression forests for conformalized quantile regression
=================================================================

An example that demonstrates the use of a quantile regression forest (QRF) to
construct reliable prediction intervals using conformalized quantile
regression (CQR). CQR offers prediction intervals that attain valid coverage,
while QRF may require additional calibration for reliable interval estimates.
Based on "Prediction intervals: Quantile Regression Forests" by Carl McBride
Ellis:
https://www.kaggle.com/code/carlmcbrideellis/prediction-intervals-quantile-regression-forests.

"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = 0
rng = check_random_state(random_state)
round_to = 3
cov_pct = 90  # the "coverage level"
alpha = (100 - cov_pct) / 100

# Load the California Housing Prices dataset.
california = datasets.fetch_california_housing()
n_samples = min(california.target.size, 1000)
perm = rng.permutation(n_samples)
X = california.data[perm]
y = california.target[perm]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


def sort_y_values(y_test, y_pred, y_pis):
    """Sort the dataset for making plots using the `fill_between` function."""
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0][indices]
    y_upper_bound = y_pis[:, 1][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def coverage_score(y_true, y_pred_low, y_pred_upp):
    coverage = np.mean((y_pred_low <= y_true) & (y_pred_upp >= y_true))
    return float(coverage)


def width_score(y_pred_low, y_pred_upp):
    mean_width = np.abs(y_pred_upp - y_pred_low).mean()
    return float(mean_width)


strategies = {
    "qrf": "Quantile Regression Forest (QRF)",
    "cqr": "Conformalized Quantile Regression (CQR)",
}


def qrf_strategy(X_train, X_test, y_train, y_test):
    qrf = RandomForestQuantileRegressor(random_state=0)
    qrf.fit(X_train, y_train)

    # Calculate lower and upper quantile values.
    y_pred_low_upp = qrf.predict(X_test, quantiles=[alpha / 2, 1 - alpha / 2])
    y_pred_low = y_pred_low_upp[:, 0]
    y_pred_upp = y_pred_low_upp[:, 1]
    y_pis = np.stack([y_pred_low, y_pred_upp], axis=1)

    # Calculate the point predictions.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    coverage = coverage_score(y_test, y_pred_low, y_pred_upp)
    width = width_score(y_pred_low, y_pred_upp)

    return coverage, width, sort_y_values(y_test, y_pred, y_pis)


def cqr_strategy(X_train, X_test, y_train, y_test):
    # Create calibration set.
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.5, random_state=random_state
    )

    qrf = RandomForestQuantileRegressor(random_state=0)
    qrf.fit(X_train, y_train)

    # Calculate lower and upper quantile values.
    y_pred_low_upp = qrf.predict(X_test, quantiles=[alpha / 2, 1 - alpha / 2])
    y_pred_low = y_pred_low_upp[:, 0]
    y_pred_upp = y_pred_low_upp[:, 1]

    # Calculate the lower and upper quantile values of the calibration set.
    y_pred_low_upp_calib = qrf.predict(X_calib, quantiles=[alpha / 2, 1 - alpha / 2])
    y_pred_low_calib = y_pred_low_upp_calib[:, 0]
    y_pred_upp_calib = y_pred_low_upp_calib[:, 1]

    a = y_pred_low_calib - y_calib
    b = y_calib - y_pred_upp_calib
    conf_scores = (np.vstack((a, b)).T).max(axis=1)

    s = np.quantile(conf_scores, (1 - alpha) * (1 + (1 / (len(y_calib)))))

    y_conf_low = y_pred_low - s
    y_conf_upp = y_pred_upp + s
    y_pis = np.stack([y_conf_low, y_conf_upp], axis=1)

    # Calculate the point predictions.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    coverage = coverage_score(y_test, y_conf_low, y_conf_upp)
    width = width_score(y_conf_low, y_conf_upp)

    return coverage, width, sort_y_values(y_test, y_pred, y_pis)


y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
coverage, width, y_sorted = {}, {}, {}
coverage["qrf"], width["qrf"], y_sorted["qrf"] = qrf_strategy(X_train, X_test, y_train, y_test)
coverage["cqr"], width["cqr"], y_sorted["cqr"] = cqr_strategy(X_train, X_test, y_train, y_test)


def plot_prediction_intervals(
    title,
    alpha,
    ax,
    y_test,
    y_pred,
    y_pred_low,
    y_pred_upp,
    coverage,
    width,
    num_plots_idx,
    price_formatter,
):
    """Plot of the prediction intervals for each method."""
    y_pred_low_ = np.take(y_pred_low, num_plots_idx)
    y_pred_upp_ = np.take(y_pred_upp, num_plots_idx)
    y_pred_ = np.take(y_pred, num_plots_idx)
    y_test_ = np.take(y_test, num_plots_idx)

    for low, mid, upp in zip(y_pred_low_, y_pred_, y_pred_upp_):
        ax.plot([mid, mid], [low, upp], lw=4, c="#e0f2ff")
    ax.plot(y_pred_, y_test_, c="#f2a619", lw=0, marker=".", ms=5)
    ax.plot(y_pred_, y_pred_low_, alpha=0.4, c="#006aff", lw=0, marker="_", ms=4)
    ax.plot(y_pred_, y_pred_upp_, alpha=0.4, c="#006aff", lw=0, marker="_", ms=4)

    ax.set_xlabel("True House Prices")
    ax.set_ylabel("Predicted House Prices")
    lims = [
        np.min(np.minimum(y_test, y_pred)),  # min of both axes
        np.max(np.maximum(y_test, y_pred)),  # max of both axes
    ]
    ax.plot(lims, lims, ls="--", lw=1, c="grey", label=None)
    at = AnchoredText(
        (
            f"PICP: {np.round(coverage, round_to)} (target = {1 - alpha})\n"
            + f"Interval Width: {np.round(width, round_to)}"
        ),
        frameon=False,
        loc=2,
    )
    ax.add_artist(at)
    ax.grid(axis="x", color="0.95")
    ax.grid(axis="y", color="0.95")
    ax.yaxis.set_major_formatter(price_formatter)
    ax.xaxis.set_major_formatter(price_formatter)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.15))

coords = [axs[0], axs[1]]
num_plots = rng.choice(len(y_test), int(len(y_test)), replace=False)
usd_formatter = FuncFormatter(lambda x, p: f"${format(int(x * 100), ',')}k")

for strategy, coord in zip(strategies.keys(), coords):
    plot_prediction_intervals(
        strategies[strategy],
        alpha,
        coord,
        *y_sorted[strategy],
        coverage[strategy],
        width[strategy],
        num_plots,
        usd_formatter,
    )

plt.subplots_adjust(top=0.15)
fig.tight_layout(pad=3)

plt.show()
