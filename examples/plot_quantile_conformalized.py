"""
=================================================================
Quantile regression forests for conformalized quantile regression
=================================================================

An example that demonstrates the use of a quantile regression forest (QRF) to
construct reliable prediction intervals using conformalized quantile
regression (CQR). CQR offers prediction intervals that attain valid coverage,
while QRF may require additional calibration for reliable interval estimates.
This example uses MAPIE to construct the CQR interval estimates with a QRF.

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
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.5, random_state=random_state
)

qrf = RandomForestQuantileRegressor(random_state=0)
qrf.fit(X_train, y_train)

# Calculate lower and upper quantile values.
y_pred_lower_upper = qrf.predict(X_test, quantiles=[alpha / 2, 1 - alpha / 2])
y_pred_lower = y_pred_lower_upper[:, 0]
y_pred_upper = y_pred_lower_upper[:, 1]

# Calculate the point predictions.
y_pred_mean = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)


def WIS_and_coverage(y_true, lower, upper, alpha):

    assert np.isnan(y_true) == False, "y_true contains NaN value(s)"
    assert np.isinf(y_true) == False, "y_true contains inf values(s)"
    assert np.isnan(lower) == False, "lower interval value contains NaN value(s)"
    assert np.isinf(lower) == False, "lower interval value contains inf values(s)"
    assert np.isnan(upper) == False, "upper interval value contains NaN value(s)"
    assert np.isinf(upper) == False, "upper interval value contains inf values(s)"
    assert alpha > 0 and alpha <= 1, f"alpha should be (0,1]. Found: {alpha}"

    # WIS for one single row
    score = np.abs(upper - lower)
    if y_true < np.minimum(upper, lower):
        score += (2 / alpha) * (np.minimum(upper, lower) - y_true)
    if y_true > np.maximum(upper, lower):
        score += (2 / alpha) * (y_true - np.maximum(upper, lower))
    # coverage for one single row
    coverage = 1  # assume is within coverage
    if (y_true < np.minimum(upper, lower)) or (y_true > np.maximum(upper, lower)):
        coverage = 0
    return score, coverage


# vectorize the function
v_WIS_and_coverage = np.vectorize(WIS_and_coverage)


def score(y_true, lower, upper, alpha):
    """
    This is an implementation of the Winkler Interval score (https://otexts.com/fpp3/distaccuracy.html#winkler-score).
    The mean over all of the individual Winkler Interval scores (MWIS) is returned, along with the coverage.

    See:
    [1] Robert L. Winkler "A Decision-Theoretic Approach to Interval Estimation", Journal of the American Statistical Association, vol. 67, pp. 187-191 (1972) (https://doi.org/10.1080/01621459.1972.10481224)
    [2] Tilmann Gneiting and Adrian E Raftery "Strictly Proper Scoring Rules, Prediction, and Estimation", Journal of the American Statistical Association, vol. 102, pp. 359-378 (2007) (https://doi.org/10.1198/016214506000001437) (Section 6.2)

    Version: 1.0.4
    Author:  Carl McBride Ellis
    Date:    2023-12-07
    """

    assert y_true.ndim == 1, "y_true: pandas Series or 1D array expected"
    assert lower.ndim == 1, "lower: pandas Series or 1D array expected"
    assert upper.ndim == 1, "upper: pandas Series or 1D array expected"
    assert isinstance(alpha, float) == True, "alpha: float expected"

    WIS_scores, coverage = v_WIS_and_coverage(y_true, lower, upper, alpha)
    MWIS = np.mean(WIS_scores)
    MWIS = float(MWIS)
    coverage = coverage.sum() / coverage.shape[0]
    coverage = float(coverage)

    return MWIS, coverage


MWIS, coverage = score(y_test, y_pred_lower, y_pred_upper, alpha)
print(f"MWI score           ", round(MWIS, 3))
print("Predictions coverage    ", round(coverage * 100, 1), "%")

# Calculate the lower and upper quantile values of the calibration set.
y_pred_lower_upper_calib = qrf.predict(X_calib, quantiles=[alpha / 2, 1 - alpha / 2])
y_pred_lower_calib = y_pred_lower_upper_calib[:, 0]
y_pred_upper_calib = y_pred_lower_upper_calib[:, 1]

a = y_pred_lower_calib - y_calib
b = y_calib - y_pred_upper_calib
conf_scores = (np.vstack((a, b)).T).max(axis=1)

s = np.quantile(conf_scores, (1 - alpha) * (1 + (1 / (len(y_calib)))))

y_conf_lower = y_pred_lower - s
y_conf_upper = y_pred_upper + s

MWIS, coverage = score(y_test, y_conf_lower, y_conf_upper, alpha)
print(f"MWI score           ", round(MWIS, 3))
print("Predictions coverage    ", round(coverage * 100, 1), "%")


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


def temp():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.15))

    coords = [axs[0], axs[1]]
    num_plots = rng.choice(len(y_test), int(len(y_test)), replace=False)
    usd_formatter = FuncFormatter(lambda x, p: f"${format(int(x * 100), ',')}k")

    for strategy, coord in zip(strategies.keys(), coords):
        plot_prediction_intervals(
            strategy,
            alpha,
            coord,
            y_test_sorted[strategy],
            y_pred_sorted[strategy],
            lower_bound[strategy],
            upper_bound[strategy],
            coverage[strategy],
            width[strategy],
            num_plots,
            usd_formatter,
        )

    plt.subplots_adjust(top=0.15)
    fig.tight_layout(pad=3)

    plt.show()
