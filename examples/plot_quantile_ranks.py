"""
Using Quantile Ranks to Identify Potential Outliers
===================================================

This example demonstrates the use of quantile regression forest (QRF) quantile
ranks to identify potential outlier samples. In this scenario, we train a QRF
model on a toy dataset and use quantile ranks to highlight values that deviate
significantly from the expected range. Potential outliers are defined as
points whose quantile rank falls outside the specified threshold interval
around the median.
"""

import math

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

random_seed = 0
n_samples = 5000
bounds = [0, 10]


def make_toy_dataset(n_samples, bounds, random_seed=0):
    rng = np.random.RandomState(random_seed)
    X_1d = np.linspace(*bounds, num=n_samples)
    X = X_1d.reshape(-1, 1)
    y = X_1d * np.cos(X_1d) + rng.normal(scale=X_1d / math.e)
    return X, y


X, y = make_toy_dataset(n_samples, bounds, random_seed=random_seed)

params = {"max_samples_leaf": None, "min_samples_leaf": 50, "random_state": random_seed}
qrf = RandomForestQuantileRegressor(**params).fit(X, y)

y_pred = qrf.predict(X, quantiles=0.5)

# Get the quantile rank for all samples.
y_ranks = qrf.quantile_ranks(X, y)

df = pd.DataFrame(
    {
        "x": X.reshape(-1),
        "y": y,
        "y_pred": y_pred,
        "y_rank": y_ranks,
    }
)


def plot_fit_and_ranks(df):
    # Slider for varying the interval that defines the upper and lower quantile rank thresholds.
    slider = alt.binding_range(min=0, max=1, step=0.01, name="Rank Interval Threshold: ")
    interval_selection = alt.param("interval_selection", bind=slider, value=0.05)

    click = alt.selection_point(fields=["outlier"], bind="legend")

    base = alt.Chart(df)

    points = (
        base.add_params(interval_selection, click)
        .transform_calculate(
            outlier="abs(datum.y_rank - 0.5) > (0.5 - interval_selection / 2) ? 'Yes' : 'No'"
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
                alt.Tooltip("x:Q", format=".3f", title="x"),
                alt.Tooltip("y:Q", format=".3f", title="f(x)"),
                alt.Tooltip("y_rank:Q", format=".3f", title="Quantile Rank"),
                alt.Tooltip("outlier:N", title="Outlier"),
            ],
        )
    )

    line_pred = base.mark_line(color="#006aff", size=4).encode(
        x=alt.X("x:Q", axis=alt.Axis(title="x")),
        y=alt.Y("y_pred:Q", axis=alt.Axis(title="f(x)")),
    )

    # For desired legend labels.
    dummy_legend = (
        base.mark_line(opacity=1)
        .encode(opacity=alt.Opacity("model:N", scale=alt.Scale(range=[1, 1]), title="Prediction"))
        .transform_calculate(model="'Median'")
    )

    chart = (dummy_legend + points + line_pred).properties(
        height=400,
        width=650,
        title="QRF Predictions with Quantile Rank Thresholding on Toy Dataset",
    )

    return chart


chart = plot_fit_and_ranks(df)
chart
