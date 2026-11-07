"""
Predicting with Quantile Regression Forests
===========================================

This example demonstrates the use of a quantile regression forest (QRF) to
predict the conditional median and construct prediction intervals. The noisy
samples are generated from the function f(x) = x sin(x) plus right-skewed
noise that grows with x, so the conditional median lies below f(x) and the
intervals are asymmetric. The QRF predictions are compared to the true
conditional median and 95% interval, which are known for this toy dataset.
Click a legend entry to highlight its marks (shift-click for more).
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy as sp
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

    sigma = 0.25 + x / 20
    noise = random_state.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f + (noise if add_noise else np.zeros_like(f))

    return np.atleast_2d(x).T, y


def get_true_quantiles(x, quantiles):
    """Get the true quantiles of the toy dataset's noisy targets at `x`."""
    sigma = 0.25 + x / 20
    z = sp.stats.norm.ppf(quantiles)  # quantiles of the lognormal noise's exponent
    return (
        x[:, None] * np.sin(x[:, None])
        + np.exp(sigma[:, None] * z)
        - np.exp(sigma**2 / 2)[:, None]
    )


# Create a noisy dataset for modeling and evenly spaced inputs for illustration.
X, y = make_toy_dataset(n_samples, bounds, add_noise=True, random_state=0)
X_func = np.linspace(*bounds, num=n_samples).reshape(-1, 1)
y_func = get_true_quantiles(X_func.ravel(), quantiles)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

qrf = RandomForestQuantileRegressor(min_samples_leaf=10, random_state=random_state)
qrf.fit(X_train, y_train)

y_pred_test = qrf.predict(X_test, quantiles=quantiles)  # predict on noisy samples
y_pred_func = qrf.predict(X_func, quantiles=quantiles)  # predict on evenly spaced inputs

df = pd.DataFrame(
    {
        "x": np.concatenate([X_test.reshape(-1), X_func.reshape(-1)]),
        "y": np.concatenate([y_test, y_func[:, 1]]),
        "y_low": np.concatenate([np.full(len(y_test), np.nan), y_func[:, 0]]),
        "y_upp": np.concatenate([np.full(len(y_test), np.nan), y_func[:, 2]]),
        "y_pred": np.concatenate([y_pred_test[:, 1], y_pred_func[:, 1]]),
        "y_pred_low": np.concatenate([y_pred_test[:, 0], y_pred_func[:, 0]]),
        "y_pred_upp": np.concatenate([y_pred_test[:, 2], y_pred_func[:, 2]]),
        "test": [True] * len(y_test) + [False] * len(y_func),
    }
)


def plot_predictions_and_intervals(df):
    """Plot model predictions and prediction intervals with ground truth."""
    colors = {
        "Test Observations": "#f2a619",
        "True Median": "gray",
        "True 95% Interval": "gray",
        "Predicted Median": "#006aff",
        "Predicted 95% Interval": "#e0f2ff",
    }
    labels = list(colors.keys())
    # The legend symbols. The lines are drawn as thin bars (dashed for the true
    # interval) rather than strokes, as stroked symbols would space the legend
    # entries unevenly.
    bar = "M-1,-0.2L1,-0.2L1,0.2L-1,0.2Z"
    dashed_bar = (
        "M-1,-0.2L-0.5,-0.2L-0.5,0.2L-1,0.2Z"
        "M-0.25,-0.2L0.25,-0.2L0.25,0.2L-0.25,0.2Z"
        "M0.5,-0.2L1,-0.2L1,0.2L0.5,0.2Z"
    )
    shapes = ["circle", bar, dashed_bar, bar, "circle"]

    # Clicking an entry of the legend highlights its marks (shift-click for more).
    click = alt.selection_point(name="click", fields=["label"], bind="legend")

    def _opacity(value):
        return alt.condition(click, alt.value(value), alt.value(0.1))

    x = alt.X("x:Q", scale=alt.Scale(nice=False), title="X")
    tooltip = [
        alt.Tooltip("x:Q", format=",.3f", title="X"),
        alt.Tooltip("y:Q", format=",.3f", title="True Median"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Median"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
    ]
    df_func = df.query("~test")  # the evenly spaced inputs

    area_pred = (
        alt.Chart(df_func.assign(label=labels[4]))
        .mark_area(color=colors[labels[4]])
        .encode(
            x=x,
            y=alt.Y("y_pred_low:Q", title="Y"),
            y2="y_pred_upp:Q",
            opacity=_opacity(0.8),
            tooltip=tooltip,
        )
    )

    circle_test = (
        alt.Chart(df.query("test").assign(label=labels[0]))
        .mark_circle(color=colors[labels[0]])
        .encode(
            x=x,
            y="y:Q",
            opacity=_opacity(0.7),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_true_interval = (
        alt.Chart(df_func.assign(label=labels[2]))
        .transform_fold(["y_low", "y_upp"], as_=["bound", "y_bound"])
        .mark_line(color=colors[labels[2]], size=2, strokeDash=[4, 4])
        .encode(
            x=x,
            y="y_bound:Q",
            detail="bound:N",
            opacity=_opacity(1),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y_low:Q", format=",.3f", title="True Lower Y"),
                alt.Tooltip("y_upp:Q", format=",.3f", title="True Upper Y"),
            ],
        )
    )

    line_pred = (
        alt.Chart(df_func.assign(label=labels[3]))
        .mark_line(color=colors[labels[3]], size=4)
        .encode(x=x, y="y_pred:Q", opacity=_opacity(1), tooltip=tooltip)
    )

    # Draw the true median over the predicted median, where they overlap.
    line_true = (
        alt.Chart(df_func.assign(label=labels[1]))
        .mark_line(color=colors[labels[1]], size=2)
        .encode(x=x, y="y:Q", opacity=_opacity(1), tooltip=tooltip)
    )

    # A legend of all of the marks, drawn by invisible points.
    legend = (
        alt.Chart(pd.DataFrame({"label": labels}))
        .mark_point(filled=True, opacity=0)
        .encode(
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=labels, range=list(colors.values())),
                legend=alt.Legend(symbolOpacity=1, rowPadding=5),
                title=None,
            ),
            shape=alt.Shape("label:N", scale=alt.Scale(domain=labels, range=shapes), title=None),
        )
        .add_params(click)
    )

    chart = alt.layer(
        area_pred, circle_test, line_true_interval, line_pred, line_true, legend
    ).properties(title="QRF Predictions vs. Ground Truth on Toy Dataset", height=400, width=650)

    return chart


chart = plot_predictions_and_intervals(df)
chart
