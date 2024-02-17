"""
Multiple-Output Quantile Regression
===================================

An example on a toy dataset that demonstrates fitting a single quantile
regressor for multiple target variables. For each target, multiple quantiles
can be estimated simultaneously.
"""

import altair as alt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)

n_samples = 2500
bounds = [0, 100]

funcs = [
    {
        "signal": lambda x: np.log1p(x + 1),
        "noise": lambda x: np.log1p(x) * np.random.uniform(size=len(x)),
    },
    {
        "signal": lambda x: np.log1p(np.sqrt(x)),
        "noise": lambda x: np.log1p(x / 2) * np.random.uniform(size=len(x)),
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


X, y = make_func_Xy(funcs, bounds, n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

qrf = RandomForestQuantileRegressor(max_samples_leaf=None, max_depth=4, random_state=0)
qrf.fit(X_train, y_train)

y_pred = qrf.predict(X, quantiles=[0.025, 0.5, 0.975], weighted_quantile=False)
y_pred = y_pred.reshape(-1, 3, len(funcs))

df = pd.DataFrame(
    {
        "x": np.tile(X.squeeze(), len(funcs)),
        "y": y.reshape(-1, order="F"),
        "y_true": np.concatenate([f["signal"](X.squeeze()) for f in funcs]),
        "y_pred": np.concatenate([y_pred[:, 1, i] for i in range(len(funcs))]),
        "y_pred_low": np.concatenate([y_pred[:, 0, i] for i in range(len(funcs))]),
        "y_pred_upp": np.concatenate([y_pred[:, 2, i] for i in range(len(funcs))]),
        "target": np.concatenate([[f"{i}"] * len(X) for i in range(len(funcs))]),
    }
)


def plot_multioutputs(df, legend):
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
    ]

    points = (
        alt.Chart(df)
        .mark_circle(color="black", opacity=0.25, size=25)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y:Q"),
            color=alt.condition(click, alt.Color("target:N"), alt.value("lightgray")),
            tooltip=tooltip,
        )
    )

    line = (
        alt.Chart(df)
        .mark_line(color="black", size=3)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False), title="x"),
            y=alt.Y("y_pred:Q", title="y"),
            color=color,
            tooltip=tooltip,
        )
    )

    area = (
        alt.Chart(df)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(nice=False), title="x"),
            y=alt.Y("y_pred_low:Q", title="y"),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=color,
            tooltip=tooltip,
        )
    )

    chart = (
        (points + area + line)
        .add_params(click)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .properties(height=400, width=650, title="Multi-target Prediction Intervals")
    )

    return chart


chart = plot_multioutputs(df, legend)
chart
