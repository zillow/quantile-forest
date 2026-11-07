"""
Multi-target Quantile Regression with QRFs
==========================================

This example demonstrates how to fit a single quantile regressor for multiple
target variables on a toy dataset. For each target, multiple quantiles can be
estimated simultaneously. In this example, the target variable has two output
values for each sample, with a single regressor used to estimate multiple
quantiles simultaneously. Three of these quantiles are visualized concurrently
for each target: the median line and the area defined by the interval points.
Use the slider to set the coverage of the interval.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 2500
bounds = [0, 100]
quantiles = np.linspace(0, 1, num=201, endpoint=True).round(3).tolist()  # in steps of 0.005

# Define functions that generate targets; each function maps to one target.
target_funcs = [
    {
        "signal": lambda x: np.log1p(x + 1),
        "noise": lambda x: np.log1p(x) * random_state.uniform(size=len(x)),
        "legend": {"0": "#f2a619"},  # plot legend value and color
    },
    {
        "signal": lambda x: np.log1p(np.sqrt(x)),
        "noise": lambda x: np.log1p(x / 2) * random_state.uniform(size=len(x)),
        "legend": {"1": "#006aff"},  # plot legend value and color
    },
]


def make_funcs_Xy(funcs, n_samples, bounds):
    """Make a dataset from specified function(s)."""
    x = np.linspace(*bounds, n_samples)
    y = np.empty((len(x), len(funcs)))
    for i, func in enumerate(funcs):
        y[:, i] = func(x)
    return np.atleast_2d(x).T, y


funcs = [lambda x, f=f: f["signal"](x) + f["noise"](x) for f in target_funcs]
legend = {k: v for f in target_funcs for k, v in f["legend"].items()}

# Create a dataset with multiple target variables.
X, y = make_funcs_Xy(funcs, n_samples, bounds)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

qrf = RandomForestQuantileRegressor(max_depth=4, max_samples_leaf=None, random_state=random_state)
qrf.fit(X_train, y_train)  # fit on all of the targets simultaneously

# Get multi-target predictions at specified quantiles, on evenly spaced inputs.
X_grid = np.linspace(*bounds, num=200).reshape(-1, 1)
y_pred = qrf.predict(X_grid, quantiles=quantiles)  # shape = (n_grid, n_targets, n_quantiles)

# The data are rounded, and the predictions are on fewer inputs than the observations, to
# keep the chart small, since it includes the predictions at every quantile.
df_obs = pd.DataFrame(
    {
        "x": np.tile(X.squeeze(), len(funcs)).round(3),
        "y": y.reshape(-1, order="F").round(3),
        "target": np.concatenate([[str(i)] * len(X) for i in range(len(funcs))]),
    }
)
df_pred = pd.DataFrame(
    {
        "x": np.tile(X_grid.squeeze(), len(funcs)).round(3),
        "y_pred": (
            np.concatenate([y_pred[:, i, len(quantiles) // 2] for i in range(len(funcs))]).round(3)
        ),
        "target": np.concatenate([[str(i)] * len(X_grid) for i in range(len(funcs))]),
        **{f"q_{q_i:.3g}": y_i.ravel().round(3) for q_i, y_i in zip(quantiles, y_pred.T)},
    }
)


def plot_multitargets(df_obs, df_pred, legend):
    """Plot predictions and prediction intervals for multi-target outputs."""
    # Slider for varying the displayed prediction intervals.
    slider = alt.binding_range(name="Prediction Interval: ", min=0, max=1, step=0.01)
    interval_val = alt.param(name="interval", value=0.95, bind=slider)

    click = alt.selection_point(bind="legend", fields=["target"], on="click")

    x = alt.X("x:Q", scale=alt.Scale(nice=False), title="X")
    color = alt.condition(
        click,
        alt.Color(
            "target:N",
            legend=alt.Legend(symbolOpacity=1),
            scale=alt.Scale(range=list(legend.values())),
            sort=list(legend.keys()),
            title="Target",
        ),
        alt.value("lightgray"),
    )

    tooltip = [
        alt.Tooltip("target:N", title="Target"),
        alt.Tooltip("x:Q", format=",.3f", title="X"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        alt.Tooltip("quantile_low:Q", format=".3f", title="Lower Quantile"),
        alt.Tooltip("quantile_upp:Q", format=".3f", title="Upper Quantile"),
    ]

    base = (
        alt.Chart(df_pred)
        .transform_calculate(
            quantile_low="round((0.5 - interval / 2) * 1000) / 1000",
            quantile_upp="round((0.5 + interval / 2) * 1000) / 1000",
            quantile_low_col="'q_' + datum.quantile_low",
            quantile_upp_col="'q_' + datum.quantile_upp",
        )
        .transform_calculate(
            y_pred_low="datum[datum.quantile_low_col]",
            y_pred_upp="datum[datum.quantile_upp_col]",
        )
    )

    circle = (
        alt.Chart(df_obs)
        .mark_circle(color="black", opacity=0.25, size=25)
        .encode(
            x=x,
            y=alt.Y("y:Q"),
            color=color,
            tooltip=[
                alt.Tooltip("target:N", title="Target"),
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
            ],
        )
    )

    area = base.mark_area(opacity=0.25).encode(
        x=x,
        y=alt.Y("y_pred_low:Q", title="Y"),
        y2=alt.Y2("y_pred_upp:Q"),
        color=color,
        tooltip=tooltip,
    )

    line = base.mark_line(color="black", size=4).encode(
        x=x,
        y=alt.Y("y_pred:Q", title="Y"),
        color=color,
        tooltip=tooltip,
    )

    chart = (
        (circle + area + line)
        .add_params(interval_val, click)
        .properties(
            title="Multi-target Predictions and Prediction Intervals on Toy Dataset",
            height=400,
            width=650,
        )
    )

    return chart


chart = plot_multitargets(df_obs, df_pred, legend)
chart
