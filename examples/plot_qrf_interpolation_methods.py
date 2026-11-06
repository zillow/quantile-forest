"""
Comparing Quantile Interpolation Methods
========================================

This example illustrates the interpolation methods that a quantile regression
forest (QRF) can use to calculate quantiles from its training samples during
prediction. The toy QRF here has one tree with one leaf, which keeps four
training samples with targets 0, 1, 2, and 3, so it predicts the same
distribution for every input. Each panel plots the quantiles that one method
predicts against the requested quantiles. The interpolating methods work from
the ranks of the samples: "linear" interpolates between neighboring samples,
while "lower", "higher", "midpoint", and "nearest" take the lower or higher
neighbor, their midpoint, or the nearer one. The inverted cumulative
distribution function (CDF) methods instead predict the smallest sample with at
least the requested fraction of the samples at or below it, and the averaged
method, the default, averages it with the next sample when the fraction is
exact (here, at 25%, 50%, and 75%), so that the median is the usual median.
Use the slider to compare the methods at a requested quantile.
"""

import altair as alt
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
quantiles = np.linspace(0, 1, num=101, endpoint=True).round(2).tolist()

# Create a simple toy dataset, whose four samples all fall in the same leaf.
X = np.zeros((4, 1))
y = np.array([0, 1, 2, 3])

# We use a single estimator that retains all leaf samples and is trained without bootstrap.
qrf = RandomForestQuantileRegressor(
    n_estimators=1,
    max_samples_leaf=None,
    bootstrap=False,
    random_state=random_state,
)
qrf.fit(X, y)

interpolations = [
    "Linear",
    "Lower",
    "Higher",
    "Midpoint",
    "Nearest",
    "Inverted CDF",
    "Averaged Inverted CDF",
]

dfs = []
for interpolation in interpolations:
    # Get predictions at each requested quantile using the specified interpolation method.
    method = interpolation.lower().replace(" ", "_")  # e.g., "averaged_inverted_cdf"
    y_pred = qrf.predict(X[:1], quantiles=quantiles, interpolation=method)

    dfs.append(
        pd.DataFrame(
            {
                "method": interpolation,
                "quantile": quantiles,
                "y_pred": y_pred[0],
            }
        )
    )
df = pd.concat(dfs, ignore_index=True)


def plot_interpolation_predictions(df, y):
    """Plot the quantiles predicted by each quantile interpolation method."""
    # Slider for varying the requested quantile, which each panel marks.
    slider = alt.binding_range(name="Requested Quantile: ", min=0, max=1, step=0.01)
    quantile_val = alt.param(name="quantile", value=0.5, bind=slider)

    filter_quantile = "round(datum.quantile * 100) == round(quantile * 100)"

    x = alt.X(
        "quantile:Q",
        axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1]),
        scale=alt.Scale(nice=False),
        title="Requested Quantile",
    )
    tooltip = [
        alt.Tooltip("method:N", title="Method"),
        alt.Tooltip("quantile:Q", format=".0%", title="Requested Quantile"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Quantile"),
    ]

    def _plot_panel(interpolation):
        df_method = df.query("method == @interpolation")

        line_pred = (
            alt.Chart(df_method)
            .mark_line(color="#006aff", size=2)
            .encode(
                x=x,
                y=alt.Y(
                    "y_pred:Q",
                    # Gridlines at the training targets, at any panel height.
                    axis=alt.Axis(tickCount=len(y), values=list(y)),
                    scale=alt.Scale(domain=[-0.25, 3.25]),
                    title="Predicted Quantile",
                ),
                tooltip=tooltip,
            )
        )

        # Mark the prediction at the requested quantile.
        rule_quantile = (
            alt.Chart(df_method)
            .transform_filter(filter_quantile)
            .mark_rule(color="gray", strokeDash=[4, 4])
            .encode(x="quantile:Q")
        )

        circle_quantile = (
            alt.Chart(df_method)
            .transform_filter(filter_quantile)
            .mark_circle(color="#006aff", opacity=1, size=75)
            .encode(x="quantile:Q", y="y_pred:Q", tooltip=tooltip)
        )

        text_quantile = (
            alt.Chart(df_method)
            .transform_filter(filter_quantile)
            .mark_text(align="right", baseline="bottom", color="gray", dx=-6, dy=-4)
            .encode(x="quantile:Q", y="y_pred:Q", text=alt.Text("y_pred:Q", format=".2f"))
        )

        title = interpolation
        if interpolation == "Averaged Inverted CDF":
            title += " (Default)"

        layers = [line_pred, rule_quantile, circle_quantile, text_quantile]
        return alt.layer(*layers).properties(title=title, height=120, width=120)

    # Plot the interpolating methods in the first row, and the inverted CDF methods in the second.
    panels = [_plot_panel(interpolation) for interpolation in interpolations]
    row_interpolating = alt.hconcat(*panels[:5])
    row_inverted_cdf = alt.hconcat(*panels[5:])

    chart = alt.vconcat(row_interpolating, row_inverted_cdf).add_params(quantile_val)
    return chart


chart = plot_interpolation_predictions(df, y)
chart
