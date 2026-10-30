"""
Using Quantile Ranks to Identify Potential Outliers
===================================================

This example demonstrates the use of quantile regression forest (QRF) quantile
ranks to identify potential outliers in a dataset. In this scenario, we train
a QRF model on a toy dataset, in which some of the targets have been shifted
to make outliers, and use quantile ranks to highlight values that deviate
significantly from the expected range. Potential outliers are defined as
points whose quantile rank falls outside the specified threshold interval
around the median. Since the ranked samples are also the training samples, we
calculate out-of-bag (OOB) quantile ranks: each sample is ranked among the
training samples of the trees that did not draw it, so that an outlier is not
ranked against itself, which would hide it. The plot shows the precision (the
percentage of potential outliers whose targets were shifted) and the recall
(the percentage of shifted targets flagged as potential outliers) at the
selected threshold.
"""

import math

import altair as alt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 5000
n_outliers = 50
bounds = [0, 10]


def make_toy_dataset(n_samples, bounds, n_outliers=0, random_state=None):
    """Make a toy dataset, with `n_outliers` outliers."""
    random_state = check_random_state(random_state)
    X_1d = np.linspace(*bounds, num=n_samples)
    X = X_1d.reshape(-1, 1)
    y = X_1d * np.cos(X_1d) + random_state.normal(scale=X_1d / math.e)

    # Shift some of the targets up or down by 5 to 10.
    outliers = random_state.choice(n_samples, n_outliers, replace=False)
    shifts = random_state.choice([-1, 1], n_outliers) * random_state.uniform(5, 10, n_outliers)
    y[outliers] += shifts
    is_outlier = np.isin(np.arange(n_samples), outliers)

    return X, y, is_outlier


# Create a toy dataset with outliers.
X, y, is_outlier = make_toy_dataset(n_samples, bounds, n_outliers=n_outliers, random_state=0)

qrf = RandomForestQuantileRegressor(
    min_samples_leaf=50,
    max_samples_leaf=None,
    random_state=random_state,
)
qrf.fit(X, y)

y_pred = qrf.predict(X, quantiles=0.5)

# Get the out-of-bag quantile rank for all samples.
y_rank = qrf.quantile_ranks(X, y, oob_score=True)  # a value in the range [0, 1] for each sample

df = pd.DataFrame(
    {
        "x": X.reshape(-1),
        "y": y,
        "y_pred": y_pred,
        "y_rank": y_rank,
        "target": np.where(is_outlier, "Shifted (Outlier)", "Original"),
    }
)


def plot_pred_and_ranks(df):
    """Plot quantile predictions and ranks."""
    # Slider for varying the interval that defines the upper and lower quantile rank thresholds.
    slider = alt.binding_range(name="Rank Interval Threshold: ", min=0, max=1, step=0.01)
    interval_val = alt.param(name="interval", value=0.01, bind=slider)

    # Clicking an entry of either legend highlights its points (double-click to reset).
    # Without toggling, shift-clicking one legend doesn't reset the other's selection.
    click = alt.selection_point(
        name="click_outlier", bind="legend", fields=["outlier"], on="click", toggle=False
    )
    click_target = alt.selection_point(
        name="click_target", bind="legend", fields=["target"], on="click", toggle=False
    )

    base = alt.Chart(df)

    # A sample is a potential outlier if its rank falls outside the threshold interval.
    is_outlier = "abs(datum.y_rank - 0.5) > (0.5 - interval / 2)"

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
        base.add_params(interval_val, click, click_target)
        .transform_calculate(
            outlier=f"{is_outlier} ? 'Yes' : 'No'",
            threshold_low="0 + interval / 2",
            threshold_upp="1 - interval / 2",
        )
        .mark_point(filled=True, opacity=0.5)
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            color=alt.condition(
                click & click_target,
                alt.Color(
                    "outlier:N",
                    scale=alt.Scale(domain=["Yes", "No"], range=["red", "#f2a619"]),
                    title="Outlier",
                ),
                alt.value("lightgray"),
            ),
            shape=alt.Shape(
                "target:N",
                scale=alt.Scale(
                    domain=["Shifted (Outlier)", "Original"], range=["diamond", "circle"]
                ),
                title="Target",
            ),
            size=alt.condition(alt.datum["target"] == "Original", alt.value(25), alt.value(80)),
            tooltip=[
                alt.Tooltip("x:Q", format=",.3f", title="X"),
                alt.Tooltip("y:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Y"),
                alt.Tooltip("y_rank:Q", format=".3f", title="Quantile Rank"),
                alt.Tooltip("threshold_low:Q", format=".3f", title="Lower Threshold"),
                alt.Tooltip("threshold_upp:Q", format=".3f", title="Upper Threshold"),
                alt.Tooltip("outlier:N", title="Outlier"),
                alt.Tooltip("target:N", title="Target"),
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

    # The precision and recall of the potential outliers at the selected threshold.
    text_metrics = (
        base.transform_calculate(
            flagged=f"{is_outlier} ? 1 : 0",
            shifted="datum.target == 'Shifted (Outlier)' ? 1 : 0",
        )
        .transform_calculate(hit="datum.flagged * datum.shifted")
        .transform_aggregate(n_flagged="sum(flagged)", n_shifted="sum(shifted)", n_hits="sum(hit)")
        .transform_calculate(
            precision=(
                "datum.n_flagged > 0 ? format(datum.n_hits / datum.n_flagged, '.1%') : 'n/a'"
            ),
            recall="format(datum.n_hits / datum.n_shifted, '.1%')",
        )
        .transform_calculate(
            text=(
                "'Precision: ' + datum.precision + ' (' + datum.n_hits + ' of '"
                " + format(datum.n_flagged, ',') + ' potential outliers)\\n'"
                " + 'Recall: ' + datum.recall + ' (' + datum.n_hits + ' of '"
                " + datum.n_shifted + ' shifted targets)'"
            )
        )
        .mark_text(align="left", baseline="top", color="gray", lineBreak="\n")
        .encode(x=alt.value(5), y=alt.value(5), text="text:N")
    )

    chart = (dummy_legend + circle + line_pred + text_metrics).properties(
        title="QRF Predictions with Quantile Rank Thresholding on Toy Dataset",
        height=400,
        width=650,
    )

    return chart


chart = plot_pred_and_ranks(df)
chart
