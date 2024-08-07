"""
Multi-target Quantile Regression with QRFs
==========================================

This example demonstrates how to fit a single quantile regressor for multiple
target variables on a toy dataset. For each target, multiple quantiles can be
estimated simultaneously. In this example, the target variable has two output
values for each sample, with a single regressor used to estimate many
quantiles simultaneously. Three of these quantiles are visualized concurrently
for each target: the median line and the area defined by the interval points.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_seed = 0
rng = check_random_state(random_seed)

n_samples = 2500
bounds = [0, 100]
quantiles = np.linspace(0, 1, num=41, endpoint=True).round(3).tolist()

# Define functions that generate targets; each function maps to one target.
funcs = [
    {
        "signal": lambda x: np.log1p(x + 1),
        "noise": lambda x: np.log1p(x) * rng.uniform(size=len(x)),
    },
    {
        "signal": lambda x: np.log1p(np.sqrt(x)),
        "noise": lambda x: np.log1p(x / 2) * rng.uniform(size=len(x)),
    },
]

legend = {
    "0": "#f2a619",
    "1": "#006aff",
}


def make_func_Xy(funcs, bounds, n_samples):
    x = np.linspace(*bounds, n_samples)
    y = np.empty((len(x), len(funcs)))
    for i, func in enumerate(funcs):
        y[:, i] = func["signal"](x) + func["noise"](x)
    return np.atleast_2d(x).T, y


def format_frac(fraction):
    formatted = ("%.3g" % fraction).rstrip("0").rstrip(".")
    return formatted if formatted else "0"


# Create the dataset with multiple target variables.
X, y = make_func_Xy(funcs, bounds, n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

qrf = RandomForestQuantileRegressor(max_samples_leaf=None, max_depth=4, random_state=random_seed)
qrf.fit(X_train, y_train)  # fit on all of the targets simultaneously

# Get multi-target predictions at specified quantiles.
y_pred = qrf.predict(X, quantiles=quantiles)

df = pd.DataFrame(
    {
        "x": np.tile(X.squeeze(), len(funcs)),
        "y": y.reshape(-1, order="F"),
        "y_true": np.concatenate([f["signal"](X.squeeze()) for f in funcs]),
        "y_pred": np.concatenate([y_pred[:, i, len(quantiles) // 2] for i in range(len(funcs))]),
        "target": np.concatenate([[f"{i}"] * len(X) for i in range(len(funcs))]),
    }
).join(
    pd.DataFrame(
        {
            f"q_{format_frac(q)}": np.concatenate([y_pred[:, t, idx] for t in range(len(funcs))])
            for idx, q in enumerate(quantiles)
        }
    )
)


def plot_multitargets(df, legend):
    # Slider for varying the displayed prediction intervals.
    slider = alt.binding_range(min=0, max=1, step=0.05, name="Prediction Interval: ")
    interval_selection = alt.param(value=0.95, bind=slider, name="interval")

    click = alt.selection_point(fields=["target"], bind="legend")

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
        alt.Tooltip("y_true:Q", format=",.3f", title="Y"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        alt.Tooltip("quantile_low:Q", format=".3f", title="Lower Quantile"),
        alt.Tooltip("quantile_upp:Q", format=".3f", title="Upper Quantile"),
    ]

    base = (
        alt.Chart(df)
        .transform_calculate(
            quantile_low=f"round((0.5 - interval / 2) * 1000) / 1000",
            quantile_upp=f"round((0.5 + interval / 2) * 1000) / 1000",
            quantile_low_col="'q_' + datum.quantile_low",
            quantile_upp_col="'q_' + datum.quantile_upp",
        )
        .transform_calculate(
            y_pred_low=f"datum[datum.quantile_low_col]",
            y_pred_upp=f"datum[datum.quantile_upp_col]",
        )
    )

    points = base.mark_circle(color="black", opacity=0.25, size=25).encode(
        x=alt.X("x:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("y:Q"),
        color=alt.condition(click, alt.Color("target:N"), alt.value("lightgray")),
        tooltip=tooltip,
    )

    line = base.mark_line(color="black", size=3).encode(
        x=alt.X("x:Q", scale=alt.Scale(nice=False), title="x"),
        y=alt.Y("y_pred:Q", title="y"),
        color=color,
        tooltip=tooltip,
    )

    area = base.mark_area(opacity=0.25).encode(
        x=alt.X("x:Q", scale=alt.Scale(nice=False), title="x"),
        y=alt.Y("y_pred_low:Q", title="y"),
        y2=alt.Y2("y_pred_upp:Q", title=None),
        color=color,
        tooltip=tooltip,
    )

    chart = (
        (points + area + line)
        .add_params(interval_selection, click)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .properties(
            height=400,
            width=650,
            title="Multi-target Predictions and Prediction Intervals on Toy Dataset",
        )
    )

    return chart


chart = plot_multitargets(df, legend)
chart
