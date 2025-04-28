"""
Using Quantile Ranks to Identify Potential Outliers
===================================================

This example demonstrates the use of quantile regression forest (QRF) quantile
ranks to identify potential outliers in a dataset. In this scenario, we train
a QRF model on a toy dataset and use quantile ranks to highlight values that
deviate significantly from the expected range. Potential outliers are defined
as points whose quantile rank falls outside the specified threshold interval
around the median.
"""

import math

import altair as alt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 5000
bounds = [0, 10]


def make_toy_dataset(n_samples, bounds, random_state=None):
    """Make a toy dataset."""
    random_state = check_random_state(random_state)
    X_1d = np.linspace(*bounds, num=n_samples)
    X = X_1d.reshape(-1, 1)
    y = X_1d * np.cos(X_1d) + random_state.normal(scale=X_1d / math.e)
    return X, y


# Create a toy dataset.
X, y = make_toy_dataset(n_samples, bounds, random_state=0)

qrf = RandomForestQuantileRegressor(
    min_samples_leaf=50,
    max_samples_leaf=None,
    random_state=random_state,
)
qrf.fit(X, y)

y_pred = qrf.predict(X, quantiles=0.5)

# Get the quantile rank for all samples.
y_rank = qrf.quantile_ranks(X, y)  # output is a value in the range [0, 1] for each sample

df = pd.DataFrame({"x": X.reshape(-1), "y": y, "y_pred": y_pred, "y_rank": y_rank})


def plot_pred_and_ranks(df):
    """Plot quantile predictions and ranks."""
    # Slider for varying the interval that defines the upper and lower quantile rank thresholds.
    slider = alt.binding_range(name="Rank Interval Threshold: ", min=0, max=1, step=0.01)
    interval_val = alt.param(name="interval", value=0.05, bind=slider)

    click = alt.selection_point(bind="legend", fields=["outlier"], on="click")

    base = alt.Chart(df)

    # For desired legend labels.
    dummy_legend = (
        base.mark_line(opacity=1)
        .encode(
            opacity=alt.Opacity(
                "model:N",
                scale=alt.Scale(range=[1, 1], domain=["Median"]),
                sort=["Median"],
                title="Prediction",
            )
        )
        .transform_calculate(model="'Median'")
    )

    circle = (
        base.add_params(interval_val, click)
        .transform_calculate(
            outlier="abs(datum.y_rank - 0.5) > (0.5 - interval / 2) ? 'Yes' : 'No'",
            threshold_low="0 + interval / 2",
            threshold_upp="1 - interval / 2",
        )
        .mark_circle(opacity=0.5, size=25)
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.condition(
                click,
                alt.Color(
                    "outlier:N",
                    scale=alt.Scale(domain=["Yes", "No"], range=["red", "#f2a619"]),
                    title="Outlier",
                ),
                alt.value("lightgray"),
            ),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
                alt.Tooltip("y_rank:Q", format=".3f", title="Quantile Rank"),
                alt.Tooltip("threshold_low:Q", format=".3f", title="Lower Threshold"),
                alt.Tooltip("threshold_upp:Q", format=".3f", title="Upper Threshold"),
                alt.Tooltip("y_rank:Q", format=".3f", title="Quantile Rank"),
                alt.Tooltip("outlier:N", title="Outlier"),
            ],
        )
    )

    line_pred = base.mark_line(color="#006aff", size=4).encode(
        x=alt.X("x:Q", axis=alt.Axis(title="X")),
        y=alt.Y("y_pred:Q", axis=alt.Axis(title="Y")),
        tooltip=[
            alt.Tooltip("x:Q", format=",.3f", title="X"),
            alt.Tooltip("y:Q", format=",.3f", title="Y"),
            alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        ],
    )

    chart = (dummy_legend + circle + line_pred).properties(
        title="QRF Predictions with Quantile Rank Thresholding on Toy Dataset",
        height=400,
        width=650,
    )

    return chart


chart = plot_pred_and_ranks(df)
chart
