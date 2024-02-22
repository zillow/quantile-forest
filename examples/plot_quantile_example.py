"""
Predicting with Quantile Regression Forests
===========================================

An example that demonstrates the use of a quantile regression forest to
predict a conditional median and prediction intervals. The example compares
the predictions to a ground truth function used to generate noisy samples.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

n_samples = 1000
bounds = [0, 10]
quantiles = [0.025, 0.5, 0.975]


def make_toy_dataset(n_samples, bounds, add_noise=True, random_seed=0):
    rng = np.random.RandomState(random_seed)

    x = rng.uniform(*bounds, size=n_samples)
    f = x * np.sin(x)

    sigma = 0.25 + x / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f + (noise if add_noise else 0)

    return np.atleast_2d(x).T, y


# Create noisy data for modeling and non-noisy function data for illustration.
X, y = make_toy_dataset(n_samples, bounds, add_noise=True, random_seed=0)
X_func, y_func = make_toy_dataset(n_samples, bounds, add_noise=False, random_seed=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

qrf = RandomForestQuantileRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
qrf.fit(X_train, y_train)

y_pred_func = qrf.predict(X_func, quantiles=quantiles)
y_pred_test = qrf.predict(X_test, quantiles=quantiles)

df = pd.DataFrame(
    {
        "X": np.concatenate([X_func.reshape(-1), X_test.reshape(-1)]),
        "y": np.concatenate([y_func, y_test]),
        "y_pred": np.concatenate([y_pred_func[:, 1], y_pred_test[:, 1]]),
        "y_pred_low": np.concatenate([y_pred_func[:, 0], y_pred_test[:, 0]]),
        "y_pred_upp": np.concatenate([y_pred_func[:, 2], y_pred_test[:, 2]]),
        "test": [False] * len(y_func) + [True] * len(y_test),
    }
)


def plot_fit_and_intervals(df):
    points = (
        alt.Chart(df.assign(**{"point_label": "Test Observations"}))
        .transform_filter(alt.datum["test"])  # filter to test data
        .mark_circle(color="#f2a619")
        .encode(
            x=alt.X("X:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y:Q", title=""),
            color=alt.Color("point_label:N", scale=alt.Scale(range=["#f2a619"]), title=None),
            tooltip=[
                alt.Tooltip("X:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_true = (
        alt.Chart(df.assign(**{"y_true_label": "f(x) = x sin(x)"}))
        .transform_filter(~alt.datum["test"])  # filter to training data
        .mark_line(color="black", size=3)
        .encode(
            x=alt.X("X:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y:Q", title="f(x)"),
            color=alt.Color("y_true_label:N", scale=alt.Scale(range=["black"]), title=None),
            tooltip=[
                alt.Tooltip("X:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_pred = (
        alt.Chart(df.assign(**{"y_pred_label": "Predicted Median"}))
        .transform_filter(~alt.datum["test"])  # filter to training data
        .mark_line(color="#006aff", size=5)
        .encode(
            x=alt.X("X:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y_pred:Q", title=""),
            color=alt.Color("y_pred_label:N", scale=alt.Scale(range=["#006aff"]), title=None),
            tooltip=[
                alt.Tooltip("X:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
            ],
        )
    )

    area_pred = (
        alt.Chart(df)
        .transform_filter(~alt.datum["test"])  # filter to training data
        .mark_area(color="#e0f2ff", opacity=0.8)
        .encode(
            x=alt.X("X:Q", scale=alt.Scale(nice=False), title="x"),
            y=alt.Y("y_pred_low:Q", title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            tooltip=[
                alt.Tooltip("X:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
                alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
                alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
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
        (area_pred + points + line_true + line_pred + blank)
        .resolve_scale(color="independent")
        .properties(height=400, width=650)
    )

    return chart


chart = plot_fit_and_intervals(df)
chart
