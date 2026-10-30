"""
Quantile Regression Forests with Sample Weights
===============================================

This example demonstrates the use of sample weights with a quantile regression
forest (QRF). A QRF weights the training samples by their sample weights both
when fitting the trees and when calculating quantiles from the training
samples in the leaves, so that a sample with weight 2 counts like two samples
with weight 1. In this scenario, a population consists of two groups, A and B,
whose targets differ, but the group of a sample is not one of its features.
Group B is 20% of the population but half of the training data, as if it had
been oversampled. Without weights, the QRF predicts the quantiles of the
oversampled training data. Weighting each training sample by its group's share
of the population divided by its share of the training data recovers the
quantiles of the population.
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.utils.validation import check_random_state

from quantile_forest import ExtraTreesQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 2000
n_test_samples = 5000
share_b_population = 0.2  # the share of group B in the population
share_b_train = 0.5  # the share of group B in the training data
quantiles = [0.1, 0.5, 0.9]


def get_signal(x):
    """Get the part of the target that the feature determines."""
    return 2 * np.sin(x) + x / 2


def make_dataset(n_samples, share_b, random_state=None):
    """Make a dataset of groups A and B, whose targets differ by 3 on average."""
    random_state = check_random_state(random_state)
    x = random_state.uniform(0, 10, size=n_samples)
    is_b = random_state.uniform(size=n_samples) < share_b
    y = get_signal(x) + 3 * is_b + random_state.normal(size=n_samples)
    return x.reshape(-1, 1), y, is_b


def get_population_quantiles(x, quantiles):
    """Get the quantiles of the population's targets at `x`."""

    def cdf(t):  # the distribution of the targets' deviations from the signal
        norm = sp.stats.norm
        return (1 - share_b_population) * norm.cdf(t) + share_b_population * norm.cdf(t - 3)

    offsets = [sp.optimize.brentq(lambda t: cdf(t) - q, -10, 10) for q in quantiles]
    return get_signal(x)[:, np.newaxis] + np.array(offsets)


X_train, y_train, is_b_train = make_dataset(n_samples, share_b_train, random_state=random_state)
X_test, y_test, _ = make_dataset(n_test_samples, share_b_population, random_state=random_state)

# Weight each training sample by its group's share of the population divided
# by its share of the training data.
sample_weight = np.where(
    is_b_train, share_b_population / share_b_train, (1 - share_b_population) / (1 - share_b_train)
)

X_grid = np.linspace(0, 10, 201).reshape(-1, 1)
y_true_grid = get_population_quantiles(X_grid[:, 0], quantiles)

dfs = []
for strategy, weights in {
    "Without Sample Weights": None,
    "With Sample Weights": sample_weight,
}.items():
    # Without bootstrapping (the default for extra trees), the samples in each
    # leaf are weighted by their sample weights.
    qrf = ExtraTreesQuantileRegressor(min_samples_leaf=10, random_state=0)
    qrf.fit(X_train, y_train, sample_weight=weights)

    y_pred_grid = qrf.predict(X_grid, quantiles=quantiles)
    y_pred_test = qrf.predict(X_test, quantiles=quantiles)

    dfs.append(
        pd.DataFrame(
            {
                "strategy": strategy,
                "x": X_grid[:, 0],
                "y_true": y_true_grid[:, 1],
                "y_true_low": y_true_grid[:, 0],
                "y_true_upp": y_true_grid[:, 2],
                "y_pred": y_pred_grid[:, 1],
                "y_pred_low": y_pred_grid[:, 0],
                "y_pred_upp": y_pred_grid[:, 2],
                # The fractions of the population below or above the predicted quantiles.
                "below_low": np.mean(y_test < y_pred_test[:, 0]),
                "below_median": np.mean(y_test < y_pred_test[:, 1]),
                "above_upp": np.mean(y_test > y_pred_test[:, 2]),
            }
        )
    )
df = pd.concat(dfs, ignore_index=True)

# A sample of the training data, for display.
df_train = pd.DataFrame(
    {
        "x": X_train[:500, 0],
        "y": y_train[:500],
        "label": np.where(is_b_train[:500], "Group B (Training)", "Group A (Training)"),
    }
)


def plot_sample_weights(df, df_train):
    """Plot predictions and prediction intervals with and without sample weights."""
    coverage = quantiles[2] - quantiles[0]
    legend = {
        "Group A (Training)": "#f2a619",
        "Group B (Training)": "#ff0251",
        "Population Median": "gray",
        f"Population {coverage:.0%} Interval": "#c0c0c0",
        "Predicted Median": "#006aff",
        f"Predicted {coverage:.0%} Interval": "#cce1ff",
    }
    labels = list(legend.keys())
    scale = alt.Scale(domain=labels, range=list(legend.values()))
    color = alt.Color("label:N", scale=scale, title=None)

    def _plot_strategy(df, strategy):
        df = df.query("strategy == @strategy")

        x = alt.X("x:Q", scale=alt.Scale(nice=False), title="X")
        tooltip = [
            alt.Tooltip("x:Q", format=",.3f", title="X"),
            alt.Tooltip("y_true:Q", format=",.3f", title="Population Median"),
            alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Median"),
            alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
            alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
        ]

        area_pred = (
            alt.Chart(df.assign(label=labels[5]))
            .mark_area(opacity=0.8)
            .encode(
                x=x,
                y=alt.Y("y_pred_low:Q", title="Y"),
                y2=alt.Y2("y_pred_upp:Q"),
                color=color,
                tooltip=tooltip,
            )
        )

        circle_train = (
            alt.Chart(df_train)
            .mark_circle(opacity=0.6, size=15)
            .encode(x=x, y=alt.Y("y:Q", title="Y"), color=color)
        )

        line_true = (
            alt.Chart(df.assign(label=labels[2]))
            .mark_line(size=2)
            .encode(x=x, y=alt.Y("y_true:Q", title="Y"), color=color, tooltip=tooltip)
        )

        line_true_interval = (
            alt.Chart(df.assign(label=labels[3]))
            .transform_fold(["y_true_low", "y_true_upp"], as_=["bound", "y_bound"])
            .mark_line(size=2, strokeDash=[4, 4])
            .encode(x=x, y=alt.Y("y_bound:Q", title="Y"), detail="bound:N", color=color)
        )

        line_pred = (
            alt.Chart(df.assign(label=labels[4]))
            .mark_line(size=3)
            .encode(x=x, y=alt.Y("y_pred:Q", title="Y"), color=color, tooltip=tooltip)
        )

        text = (
            alt.Chart(df.head(1))
            .transform_calculate(
                median_text=(
                    "'Below Median: ' + format(datum.below_median, '.1%') + ' (target = 50%)'"
                ),
                low_text=(
                    f"'Below {quantiles[0]:.0%} Quantile: ' + format(datum.below_low, '.1%')"
                    f" + ' (target = {quantiles[0]:.0%})'"
                ),
                upp_text=(
                    f"'Above {quantiles[2]:.0%} Quantile: ' + format(datum.above_upp, '.1%')"
                    f" + ' (target = {1 - quantiles[2]:.0%})'"
                ),
            )
            .mark_text(align="left", baseline="top", color="gray")
            .encode(x=alt.value(5))
        )
        text = (
            text.encode(y=alt.value(5), text=alt.Text("median_text:N"))
            + text.encode(y=alt.value(20), text=alt.Text("low_text:N"))
            + text.encode(y=alt.value(35), text=alt.Text("upp_text:N"))
        )

        chart = (
            area_pred + circle_train + line_true_interval + line_true + line_pred + text
        ).properties(title=strategy, height=300, width=325)
        return chart

    chart = alt.hconcat(*[_plot_strategy(df, strategy) for strategy in df["strategy"].unique()])
    return chart


chart = plot_sample_weights(df, df_train)
chart
