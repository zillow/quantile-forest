"""
Multi-target Quantile Regression with QRFs
==========================================

This example demonstrates how to fit a single quantile regressor for multiple
target variables on a toy dataset. For each target, multiple quantiles can be
estimated simultaneously. In this example, the target variable has two output
values for each sample, with a single regressor used to estimate multiple
quantiles simultaneously. Three of these quantiles are visualized concurrently
for each target: the median line and the area defined by the interval points.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 2500
bounds = [0, 100]
quantiles = np.linspace(0, 1, num=41, endpoint=True).round(3).tolist()

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

# Get multi-target predictions at specified quantiles.
y_pred = qrf.predict(X, quantiles=quantiles)  # output shape = (n_samples, n_targets, n_quantiles)

df = pd.DataFrame(
    {
        "x": np.tile(X.squeeze(), len(funcs)),
        "y": y.reshape(-1, order="F"),
        "y_pred": np.concatenate([y_pred[:, i, len(quantiles) // 2] for i in range(len(funcs))]),
        "target": np.concatenate([[str(i)] * len(X) for i in range(len(funcs))]),
        **{f"q_{q_i:.3g}": y_i.ravel() for q_i, y_i in zip(quantiles, y_pred.T)},
    }
)


def plot_multitargets(df, legend):
    """Plot predictions and prediction intervals for multi-target outputs."""
    # Slider for varying the displayed prediction intervals.
    slider = alt.binding_range(name="Prediction Interval: ", min=0, max=1, step=0.05)
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
        alt.Tooltip("y:Q", format=",.3f", title="Y"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        alt.Tooltip("quantile_low:Q", format=".3f", title="Lower Quantile"),
        alt.Tooltip("quantile_upp:Q", format=".3f", title="Upper Quantile"),
    ]

    base = (
        alt.Chart(df)
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

    circle = base.mark_circle(color="black", opacity=0.25, size=25).encode(
        x=x,
        y=alt.Y("y:Q"),
        color=color,
        tooltip=tooltip,
    )

    area = base.mark_area(opacity=0.25).encode(
        x=x,
        y=alt.Y("y_pred_low:Q", title="Y"),
        y2=alt.Y2("y_pred_upp:Q"),
        color=color,
        tooltip=tooltip,
    )

    line = base.mark_line(color="black", size=3).encode(
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


chart = plot_multitargets(df, legend)
chart
