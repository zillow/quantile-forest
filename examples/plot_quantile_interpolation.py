"""
Comparing Quantile Interpolation Methods
========================================

This example illustrates the interpolation methods that can be applied during
prediction when the desired quantile lies between two data points. In this toy
example, the forest estimator creates a single split that separates samples
1–3 and samples 4–5, with quantiles calculated separately for these two groups
based on the actual sample values. The interpolation methods are used when a
calculated quantile does not precisely correspond to one of the actual values.
"""

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

intervals = list(np.arange(101) / 100)

# Create toy dataset.
X = np.array([[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1]])
y = np.array([-2, -1, 0, 1, 2])

qrf = RandomForestQuantileRegressor(
    n_estimators=1,
    max_samples_leaf=None,
    bootstrap=False,
    random_state=0,
)
qrf.fit(X, y)

interpolations = {
    "Linear": "#006aff",
    "Lower": "#ffd237",
    "Higher": "#0d4599",
    "Midpoint": "#f2a619",
    "Nearest": "#a6e5ff",
}

# legend = {"Actual": "#000000"} | interpolations
legend = {"Actual": "#000000"}
legend.update(interpolations)

dfs = []
for idx, interval in enumerate(intervals):
    # Initialize data with actual values.
    data = {
        "method": ["Actual"] * len(y),
        "X": [f"Sample {idx + 1} ({x})" for idx, x in enumerate(X.tolist())],
        "y_pred": y.tolist(),
        "y_pred_low": y.tolist(),
        "y_pred_upp": y.tolist(),
        "quantile_low": [None] * len(y),
        "quantile_upp": [None] * len(y),
    }

    # Populate data based on prediction results with different interpolations.
    for interpolation in interpolations:
        # Get predictions at median and prediction intervals.
        quantiles = [0.5, round(0.5 - interval / 2, 3), round(0.5 + interval / 2, 3)]
        y_pred = qrf.predict(X, quantiles=quantiles, interpolation=interpolation.lower())

        data["method"].extend([interpolation] * len(y))
        data["X"].extend([f"Sample {idx + 1} ({x})" for idx, x in enumerate(X.tolist())])
        data["y_pred"].extend(y_pred[:, 0])
        data["y_pred_low"].extend(y_pred[:, 1])
        data["y_pred_upp"].extend(y_pred[:, 2])
        data["quantile_low"].extend([quantiles[1]] * len(y))
        data["quantile_upp"].extend([quantiles[2]] * len(y))

    df_i = pd.DataFrame(data)
    dfs.append(df_i)
df = pd.concat(dfs)


def plot_interpolations(df, legend):
    slider = alt.binding_range(min=0, max=1, step=0.01, name="Prediction Interval: ")
    interval_selection = alt.param(value=0.9, bind=slider, name="interval")
    interval_tol = 0.001

    click = alt.selection_point(fields=["method"], bind="legend")

    color = alt.condition(
        click,
        alt.Color("method:N", sort=list(legend.keys()), title=None),
        alt.value("lightgray"),
    )

    tooltip = [
        alt.Tooltip("method:N", title="Method"),
        alt.Tooltip("X:N", title="X Values"),
        alt.Tooltip("y_pred:Q", format=".3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=".3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=".3f", title="Predicted Upper Y"),
        alt.Tooltip("quantile_low:Q", format=".3f", title="Lower Quantile"),
        alt.Tooltip("quantile_upp:Q", format=".3f", title="Upper Quantile"),
    ]

    point = (
        alt.Chart(df, width=alt.Step(20))
        .mark_circle(opacity=1, size=75)
        .encode(
            x=alt.X(
                "method:N",
                axis=alt.Axis(labels=False, tickSize=0),
                sort=list(legend.keys()),
                title=None,
            ),
            y=alt.Y("y_pred:Q", title="Actual and Predicted Values"),
            color=color,
            tooltip=tooltip,
        )
    )

    area = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "method:N",
                axis=alt.Axis(labels=False, tickSize=0),
                sort=list(legend.keys()),
                title=None,
            ),
            y=alt.Y("y_pred_low:Q", title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=color,
            tooltip=tooltip,
        )
    )

    chart = (
        (area + point)
        .transform_filter(
            (
                (alt.datum.quantile_low >= (0.5 - interval_selection / 2 - interval_tol))
                & (alt.datum.quantile_low <= (0.5 - interval_selection / 2 + interval_tol))
            )
            | (
                (alt.datum.quantile_upp >= (0.5 + interval_selection / 2 - interval_tol))
                & (alt.datum.quantile_upp <= (0.5 + interval_selection / 2 + interval_tol))
            )
            | (alt.datum.method == "Actual")
        )
        .add_params(interval_selection, click)
        .properties(height=400)
        .facet(
            column=alt.Column(
                "X:N",
                header=alt.Header(labelOrient="bottom", titleOrient="bottom"),
                title="Samples (Feature Values)",
            )
        )
        .configure_facet(spacing=15)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .configure_scale(bandPaddingInner=0.9)
        .configure_view(stroke=None)
    )

    return chart


chart = plot_interpolations(df, legend)
chart
