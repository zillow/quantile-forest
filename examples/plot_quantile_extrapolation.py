"""
Extrapolation with Quantile Regression Forests
==============================================

An example on a toy dataset that demonstrates that the prediction intervals
produced by a quantile regression forest do not extrapolate outside of the
bounds of the data in the training set, an important limitation of the
approach.
"""

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)

n_samples = 1000
bounds = [0, 15]
extrap_pct = 0.2
func = lambda x: x * np.sin(x)


def make_func_Xy(func, bounds, n_samples):
    x = np.linspace(bounds[0], bounds[1], n_samples)
    f = func(x)
    std = 0.01 + np.abs(x - 5.0) / 5.0
    noise = np.random.normal(scale=std)
    y = f + noise
    return x, y


def get_train_Xy(X, y, min_idx, max_idx):
    X_train = X[min_idx:max_idx]
    y_train = y[min_idx:max_idx]
    return X_train, y_train


def get_test_X(X):
    n_samples = len(X)
    X_test = np.atleast_2d(np.linspace(*bounds, n_samples)).T
    return X_test


X, y = make_func_Xy(func, bounds, n_samples)

extrap_min_idx = int(n_samples * (extrap_pct / 2))
extrap_max_idx = int(n_samples - (n_samples * (extrap_pct / 2)))

X_train, y_train = get_train_Xy(X, y, extrap_min_idx, extrap_max_idx)
X_test = get_test_X(X)

qrf = RandomForestQuantileRegressor(
    max_samples_leaf=None,
    min_samples_leaf=10,
    random_state=0,
)
qrf.fit(np.expand_dims(X_train, axis=-1), y_train)

y_pred = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])  # extrapolate


df = pd.DataFrame(
    {
        "X_true": X,
        "y_func": func(X),
        "y_true": y,
        "y_pred": y_pred[:, 1],
        "y_pred_low": y_pred[:, 0],
        "y_pred_upp": y_pred[:, 2],
        "train": np.concatenate(
            [
                np.zeros(extrap_min_idx),
                np.ones(extrap_max_idx - extrap_min_idx),
                np.zeros(len(y) - extrap_max_idx),
            ]
        ),
        "test_left": np.concatenate(
            [
                np.ones(extrap_min_idx),
                np.zeros(len(y) - extrap_min_idx),
            ]
        ),
        "test_right": np.concatenate(
            [
                np.zeros(extrap_max_idx),
                np.ones(len(y) - extrap_max_idx),
            ]
        ),
    }
)


def plot_extrapolations(df, x_domain=None, title=""):
    x_scale = None
    if x_domain is not None:
        x_scale = alt.Scale(domain=x_domain)

    tooltip_true = [
        alt.Tooltip("X_true:Q", format=",.3f", title="X"),
        alt.Tooltip("y_true:Q", format=",.3f", title="Y"),
    ]

    tooltip_pred = tooltip_true + [
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
    ]

    base = alt.Chart(df)

    points_true = base.mark_circle().encode(
        x=alt.X("X_true:Q", scale=x_scale, title="x"),
        y=alt.Y("y_true:Q", title="y"),
        color=alt.value("#f2a619"),
        tooltip=tooltip_true,
    )

    line_true = base.mark_line().encode(
        x=alt.X("X_true:Q", scale=x_scale, title=""),
        y=alt.Y("y_func:Q", title=""),
        color=alt.value("black"),
        tooltip=tooltip_true,
    )

    line_pred = base.mark_line().encode(
        x=alt.X("X_true:Q", scale=x_scale, title=""),
        y=alt.Y("y_pred:Q", title=""),
        color=alt.condition(alt.datum["extrapolate"], alt.value("red"), alt.value("#006aff")),
        tooltip=tooltip_pred,
    )

    bar_pred = base.mark_bar(width=2).encode(
        x=alt.X("X_true:Q", scale=x_scale, title=""),
        y=alt.Y("y_pred_low:Q", title=""),
        y2=alt.Y2("y_pred_upp:Q", title=None),
        color=alt.condition(alt.datum["extrapolate"], alt.value("red"), alt.value("#e0f2ff")),
        opacity=alt.condition(alt.datum["extrapolate"], alt.value(0.05), alt.value(0.8)),
        tooltip=tooltip_pred,
    )

    base1 = bar_pred + points_true + line_true + line_pred

    chart = base1.properties(height=250, width=375, title=title)
    return chart


kwargs = {"x_domain": [int(np.min(df["X_true"])), int(np.max(df["X_true"]))]}

chart1 = plot_extrapolations(
    df.query(f"train == 1"), title="Prediction Intervals on Training Data", **kwargs
)
chart2 = alt.layer(
    plot_extrapolations(
        df.query(f"(train == 1)"),
        title="Prediction Intervals with Extrapolated Values",
        **kwargs,
    ),
    plot_extrapolations(df.query(f"(test_left == 1)").assign(extrapolate=True), **kwargs),
    plot_extrapolations(df.query(f"(test_right == 1)").assign(extrapolate=True), **kwargs),
)
chart = chart1 | chart2
chart
