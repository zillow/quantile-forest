"""
Comparing Quantile Interpolation Methods
========================================

An example comparison of interpolation methods that can be applied during
prediction when the desired quantile lies between two data points.
"""

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

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

# Create toy dataset.
X = np.array([[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1]])
y = np.array([-2, -1, 0, 1, 2])

est = RandomForestQuantileRegressor(
    n_estimators=1,
    max_samples_leaf=None,
    bootstrap=False,
    random_state=0,
)
est.fit(X, y)

y_medians = []
y_errs = []
for interpolation in interpolations:
    y_pred = est.predict(
        X,
        quantiles=[0.025, 0.5, 0.975],
        interpolation=interpolation.lower(),
    )
    y_medians.append(y_pred[:, 1])
    y_errs.append(
        np.concatenate(
            (
                [y_pred[:, 1] - y_pred[:, 0]],
                [y_pred[:, 2] - y_pred[:, 1]],
            ),
            axis=0,
        )
    )

data = {
    "method": ["Actual"] * len(y),
    "x": [f"Sample {idx + 1} ({x})" for idx, x in enumerate(X.tolist())],
    "y_med": y.tolist(),
    "y_low": y.tolist(),
    "y_upp": y.tolist(),
}
for idx, interpolation in enumerate(interpolations):
    data["method"].extend([interpolation] * len(y))
    data["x"].extend([f"Sample {idx + 1} ({x})" for idx, x in enumerate(X.tolist())])
    data["y_med"].extend(y_medians[idx])
    data["y_low"].extend(y_medians[idx] - y_errs[idx][0])
    data["y_upp"].extend(y_medians[idx] + y_errs[idx][1])

df = pd.DataFrame(data)


def plot_interpolations(df, legend):
    click = alt.selection_point(fields=["method"], bind="legend")

    color = alt.condition(
        click,
        alt.Color("method:N", sort=list(legend.keys()), title=None),
        alt.value("lightgray"),
    )

    tooltip = [
        alt.Tooltip("method:N", title="Method"),
        alt.Tooltip("x:N", title="X Values"),
        alt.Tooltip("y_med:N", format=".3f", title="Median Y Value"),
        alt.Tooltip("y_low:N", format=".3f", title="Lower Y Value"),
        alt.Tooltip("y_upp:N", format=".3f", title="Upper Y Value"),
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
            y=alt.Y("y_med:Q", title="Actual and Predicted Values"),
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
            y=alt.Y("y_low:Q", title=""),
            y2=alt.Y2("y_upp:Q", title=None),
            color=color,
            tooltip=tooltip,
        )
    )

    chart = (
        (area + point)
        .add_params(click)
        .properties(height=400)
        .facet(
            column=alt.Column(
                "x:N",
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
