"""
Comparing Quantile Interpolation Methods
========================================

This example illustrates different interpolation methods that can be used
during prediction in quantile regression forests (QRF). When a desired quantile
lies between two data points, interpolation methods determine the predicted
value. In this toy example, the QRF creates a split that divides the samples
into two groups (samples 1–3 and samples 4–5), with quantiles calculated
separately for each. The interpolation methods demonstrate how predictions are
handled when a quantile does not exactly match a data point.

"""

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
intervals = np.linspace(0, 1, num=101, endpoint=True).round(2).tolist()

# Create a simple toy dataset.
X = np.array([[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1]])
y = np.array([-2, -1, 0, 1, 2])

# We use a single estimator that retains all leaf samples and is trained without bootstrap.
# By construction of the data, this leads to samples split between two terminal leaf nodes.
qrf = RandomForestQuantileRegressor(
    n_estimators=1,
    max_samples_leaf=None,
    bootstrap=False,
    random_state=random_state,
)
qrf.fit(X, y)

interpolations = {
    "Linear": "#006aff",
    "Lower": "#ffd237",
    "Higher": "#0d4599",
    "Midpoint": "#f2a619",
    "Nearest": "#a6e5ff",
}

legend = {"Actual": "#808080"}
legend.update(interpolations)

dfs = []
for idx, interval in enumerate(intervals):
    # Initialize data with actual values.
    data = {
        "method": ["Actual"] * len(y),
        "X": [f"Sample {idx + 1} ({x})" for idx, x in enumerate(X.tolist())],
        "y_pred": y.tolist(),
        "y_pred_low": [None] * len(y),
        "y_pred_upp": [None] * len(y),
        "quantile_low": [None] * len(y),
        "quantile_upp": [None] * len(y),
    }

    # Make predictions at the median and intervals.
    quantiles = [0.5, round(0.5 - interval / 2, 3), round(0.5 + interval / 2, 3)]

    # Populate data based on prediction results with different interpolations.
    for interpolation in interpolations:
        # Get predictions using the specified quantiles and interpolation method.
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
df = pd.concat(dfs, ignore_index=True)


def plot_interpolation_predictions(df, legend):
    """Plot predictions by quantile interpolation methods."""
    # Slider for varying the prediction interval that determines the quantiles being interpolated.
    slider = alt.binding_range(name="Prediction Interval: ", min=0, max=1, step=0.01)
    interval_val = alt.param(name="interval", value=0.8, bind=slider)

    click = alt.selection_point(bind="legend", fields=["method"], on="click")

    color = alt.condition(
        click,
        alt.Color("method:N", sort=list(legend.keys()), title=None),
        alt.value("lightgray"),
    )

    tooltip = [
        alt.Tooltip("method:N", title="Method"),
        alt.Tooltip("X:N", title="X Values"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        alt.Tooltip("quantile_low:Q", format=".3f", title="Lower Quantile"),
        alt.Tooltip("quantile_upp:Q", format=".3f", title="Upper Quantile"),
    ]

    bar_pred = (
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

    circle_pred = (
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

    chart = (
        (bar_pred + circle_pred)
        .add_params(interval_val, click)
        .transform_filter(
            "(datum.method == 'Actual')"
            "| (datum.quantile_low == round((0.5 - interval / 2) * 1000) / 1000)"
            "| (datum.quantile_upp == round((0.5 + interval / 2) * 1000) / 1000)"
        )
        .properties(height=400)
        .facet(
            column=alt.Column(
                "X:N",
                header=alt.Header(labelOrient="bottom", titleOrient="bottom"),
                title="Samples (Feature Values)",
            ),
            title="QRF Predictions by Quantile Interpolation on Toy Dataset",
        )
        .configure_facet(spacing=15)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .configure_scale(bandPaddingInner=0.9)
        .configure_title(anchor="middle")
        .configure_view(stroke=None)
    )

    return chart


chart = plot_interpolation_predictions(df, legend)
chart
