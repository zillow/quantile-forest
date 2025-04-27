"""
Predicting with Quantile Regression Forests
===========================================

This example demonstrates the use of a quantile regression forest (QRF) to
predict the conditional median and construct prediction intervals. The QRF
predictions are compared to the ground truth function used to generate noisy
samples.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 1000
bounds = [0, 10]
quantiles = [0.025, 0.5, 0.975]


def make_toy_dataset(n_samples, bounds, add_noise=True, random_state=None):
    """Make a toy dataset."""
    random_state = check_random_state(random_state)

    x = random_state.uniform(*bounds, size=n_samples)
    f = x * np.sin(x)

    sigma = 0.25 + x / 10
    noise = random_state.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f + (noise if add_noise else np.zeros_like(f))

    return np.atleast_2d(x).T, y


# Create a noisy dataset for modeling and a non-noisy version for illustration.
X, y = make_toy_dataset(n_samples, bounds, add_noise=True, random_state=0)
X_func, y_func = make_toy_dataset(n_samples, bounds, add_noise=False, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

qrf = RandomForestQuantileRegressor(max_depth=3, min_samples_leaf=5, random_state=random_state)
qrf.fit(X_train, y_train)

y_pred_test = qrf.predict(X_test, quantiles=quantiles)  # predict on noisy samples
y_pred_func = qrf.predict(X_func, quantiles=quantiles)  # predict on non-noisy samples

df = pd.DataFrame(
    {
        "x": np.concatenate([X_test.reshape(-1), X_func.reshape(-1)]),
        "y": np.concatenate([y_test, y_func]),
        "y_pred": np.concatenate([y_pred_test[:, 1], y_pred_func[:, 1]]),
        "y_pred_low": np.concatenate([y_pred_test[:, 0], y_pred_func[:, 0]]),
        "y_pred_upp": np.concatenate([y_pred_test[:, 2], y_pred_func[:, 2]]),
        "test": [True] * len(y_test) + [False] * len(y_func),
    }
)


def plot_predictions_and_intervals(df):
    """Plot model predictions and prediction intervals with ground truth."""
    area_pred = (
        alt.Chart(df)
        .transform_filter(~alt.datum["test"])  # filter to non-test data
        .mark_area(color="#e0f2ff", opacity=0.8)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False), title="X"),
            y=alt.Y("y_pred_low:Q", title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
                alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
                alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
            ],
        )
    )

    circle_test = (
        alt.Chart(df.assign(**{"point_label": "Test Observations"}))
        .transform_filter(alt.datum["test"])  # filter to test data
        .mark_circle(color="#f2a619")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y:Q", title=""),
            color=alt.Color("point_label:N", scale=alt.Scale(range=["#f2a619"]), title=None),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_true = (
        alt.Chart(df.assign(**{"y_true_label": "f(x) = x sin(x)"}))
        .transform_filter(~alt.datum["test"])  # filter to non-test data
        .mark_line(color="gray", size=3)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y:Q", title="Y"),
            color=alt.Color("y_true_label:N", scale=alt.Scale(range=["gray"]), title=None),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_pred = (
        alt.Chart(df.assign(**{"y_pred_label": "Predicted Median"}))
        .transform_filter(~alt.datum["test"])  # filter to non-test data
        .mark_line(color="#006aff", size=5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y_pred:Q", title=""),
            color=alt.Color("y_pred_label:N", scale=alt.Scale(range=["#006aff"]), title=None),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
            ],
        )
    )

    # For desired legend ordering.
    blank = (
        alt.Chart(pd.DataFrame({"y_area_label": ["Predicted 95% Interval"]}))
        .mark_area(color="#e0f2ff")
        .encode(
            color=alt.Color("y_area_label:N", scale=alt.Scale(range=["#e0f2ff"]), title=None),
        )
    )

    chart = (
        (area_pred + circle_test + line_true + line_pred + blank)
        .resolve_scale(color="independent")
        .properties(title="QRF Predictions vs. Ground Truth on Toy Dataset", height=400, width=650)
    )

    return chart


chart = plot_predictions_and_intervals(df)
chart
