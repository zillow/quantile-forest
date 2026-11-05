"""
Quantile Regression Forests with Sample Weights
===============================================

This example demonstrates the use of sample weights with a quantile regression
forest (QRF), which weights the training samples both when fitting the trees
and when calculating quantiles from the samples in the leaves, so that a sample
with weight 2 counts like two with weight 1. Here, a population consists of two
groups, A and B, whose targets differ, but a sample's group is not one of its
features. Group B is 20% of the population but half of the training data, as if
oversampled, so without weights, the QRF predicts the quantiles of the training
data. Weighting each training sample by its group's share of the population
divided by its share of the training data recovers the population's quantiles.
Use the slider to set the share of group B that the weights assume (the
population stays the same): at 20%, its true share, they recover the
population's quantiles; at 50%, its share of the training data, they are all 1,
as without weights; and at 0%, they leave group B out. The area of each
training sample shows its weight.
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
shares_b = np.linspace(0, share_b_train, num=11).round(2).tolist()  # the shares the weights assume
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

X_grid = np.linspace(0, 10, 201).reshape(-1, 1)
y_true_grid = get_population_quantiles(X_grid[:, 0], quantiles)
df_true = pd.DataFrame(
    {
        "x": X_grid[:, 0],
        "y_true": y_true_grid[:, 1],
        "y_true_low": y_true_grid[:, 0],
        "y_true_upp": y_true_grid[:, 2],
    }
)

dfs, dfs_metrics = [], []
for share_b in shares_b:
    # Weight each training sample by its group's assumed share of the population
    # divided by its share of the training data.
    sample_weight = np.where(
        is_b_train, share_b / share_b_train, (1 - share_b) / (1 - share_b_train)
    )

    # Without bootstrapping (the default for extra trees), the samples in each
    # leaf are weighted by their sample weights.
    qrf = ExtraTreesQuantileRegressor(min_samples_leaf=10, random_state=0)
    qrf.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred_grid = qrf.predict(X_grid, quantiles=quantiles)
    y_pred_test = qrf.predict(X_test, quantiles=quantiles)

    dfs.append(
        pd.DataFrame(
            {
                "share_b": share_b,
                "x": X_grid[:, 0],
                "y_pred": y_pred_grid[:, 1],
                "y_pred_low": y_pred_grid[:, 0],
                "y_pred_upp": y_pred_grid[:, 2],
            }
        )
    )

    # The fractions of the population below or above the predicted quantiles.
    dfs_metrics.append(
        pd.DataFrame(
            {
                "share_b": [share_b],
                "below_low": [np.mean(y_test < y_pred_test[:, 0])],
                "below_median": [np.mean(y_test < y_pred_test[:, 1])],
                "above_upp": [np.mean(y_test > y_pred_test[:, 2])],
            }
        )
    )
df = pd.concat(dfs, ignore_index=True)
df_metrics = pd.concat(dfs_metrics, ignore_index=True)

# A sample of the training data, for display.
df_train = pd.DataFrame(
    {
        "x": X_train[:500, 0],
        "y": y_train[:500],
        "label": np.where(is_b_train[:500], "Group B (Training)", "Group A (Training)"),
    }
)


def plot_sample_weights(df, df_true, df_train, df_metrics):
    """Plot predictions and prediction intervals with and without sample weights."""
    coverage = quantiles[2] - quantiles[0]
    colors = {
        "Group A (Training)": "#f2a619",
        "Group B (Training)": "red",
        "Population Median": "gray",
        f"Population {coverage:.0%} Interval": "gray",
        "Predicted Median": "#006aff",
        f"Predicted {coverage:.0%} Interval": "#e0f2ff",
    }
    labels = list(colors.keys())
    # The legend symbols. The lines are drawn as thin bars (dashed for the population
    # interval) rather than strokes, as stroked symbols would space the legend entries
    # unevenly.
    bar = "M-1,-0.2L1,-0.2L1,0.2L-1,0.2Z"
    dashed_bar = (
        "M-1,-0.2L-0.5,-0.2L-0.5,0.2L-1,0.2Z"
        "M-0.25,-0.2L0.25,-0.2L0.25,0.2L-0.25,0.2Z"
        "M0.5,-0.2L1,-0.2L1,0.2L0.5,0.2Z"
    )
    shapes = ["circle", "circle", bar, dashed_bar, bar, "circle"]

    # Slider for varying the share of group B in the population that the weights assume.
    slider = alt.binding_range(
        name="Share of Group B Assumed by the Weights: ", min=0, max=share_b_train, step=0.05
    )
    share_val = alt.param(name="share_b", value=share_b_population, bind=slider)

    # Clicking an entry of the legend highlights its marks (shift-click for more).
    click = alt.selection_point(name="click", fields=["label"], bind="legend")

    def _opacity(value):
        return alt.condition(click, alt.value(value), alt.value(0.1))

    def _text(metrics):
        low, upp = quantiles[0], quantiles[2]
        return "\n".join(
            [
                f"Below Median: {metrics['below_median']:.1%} (target = 50%)",
                f"Below {low:.0%} Quantile: {metrics['below_low']:.1%} (target = {low:.0%})",
                f"Above {upp:.0%} Quantile: {metrics['above_upp']:.1%} "
                f"(target = {1 - upp:.0%})",
            ]
        )

    df_text = df_metrics.assign(text=df_metrics.apply(_text, axis=1))

    x = alt.X("x:Q", scale=alt.Scale(nice=False), title="X")
    y_scale = alt.Scale(domain=[-2, 14])  # with room for the metrics above the data
    tooltip = [
        alt.Tooltip("x:Q", format=",.3f", title="X"),
        alt.Tooltip("y_pred:Q", format=",.3f", title="Predicted Median"),
        alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
        alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
    ]

    def _plot_panel(title, share, legend=False):
        # The predictions and the weights for the share of group B that the weights
        # assume, given as an expression.
        filter_share = f"round(datum.share_b * 100) == round({share} * 100)"
        weights = {
            labels[0]: f"(1 - {share}) / (1 - {share_b_train})",
            labels[1]: f"{share} / {share_b_train}",
        }

        area_pred = (
            alt.Chart(df.assign(label=labels[5]))
            .transform_filter(filter_share)
            .mark_area(color=colors[labels[5]])
            .encode(
                x=x,
                y=alt.Y("y_pred_low:Q", scale=y_scale, title="Y"),
                y2="y_pred_upp:Q",
                opacity=_opacity(0.8),
                tooltip=tooltip,
            )
        )

        circles_train = [
            alt.Chart(df_train[df_train["label"] == label])
            .transform_calculate(weight=weights[label])
            .mark_circle(clip=True, color=colors[label])
            .encode(
                x=x,
                y="y:Q",
                opacity=_opacity(0.6),
                size=alt.Size(
                    "weight:Q",
                    scale=alt.Scale(domain=[0, 2], range=[0, 30]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("x:Q", format=",.3f", title="X"),
                    alt.Tooltip("y:Q", format=",.3f", title="Y"),
                    alt.Tooltip("weight:Q", format=".2f", title="Weight"),
                ],
            )
            for label in labels[:2]
        ]

        line_true_interval = (
            alt.Chart(df_true.assign(label=labels[3]))
            .transform_fold(["y_true_low", "y_true_upp"], as_=["bound", "y_bound"])
            .mark_line(color=colors[labels[3]], size=2, strokeDash=[4, 4])
            .encode(
                x=x,
                y="y_bound:Q",
                detail="bound:N",
                opacity=_opacity(1),
                tooltip=[
                    alt.Tooltip("x:Q", format=",.3f", title="X"),
                    alt.Tooltip("y_true_low:Q", format=",.3f", title="Population Lower Y"),
                    alt.Tooltip("y_true_upp:Q", format=",.3f", title="Population Upper Y"),
                ],
            )
        )

        line_pred = (
            alt.Chart(df.assign(label=labels[4]))
            .transform_filter(filter_share)
            .mark_line(color=colors[labels[4]], size=4)
            .encode(x=x, y="y_pred:Q", opacity=_opacity(1), tooltip=tooltip)
        )

        # Draw the population median over the predicted median, where they overlap.
        line_true = (
            alt.Chart(df_true.assign(label=labels[2]))
            .mark_line(color=colors[labels[2]], size=2)
            .encode(
                x=x,
                y="y_true:Q",
                opacity=_opacity(1),
                tooltip=[
                    alt.Tooltip("x:Q", format=",.3f", title="X"),
                    alt.Tooltip("y_true:Q", format=",.3f", title="Population Median"),
                ],
            )
        )

        text_metrics = (
            alt.Chart(df_text)
            .transform_filter(filter_share)
            .mark_text(align="left", baseline="top", color="gray", lineBreak="\n")
            .encode(x=alt.value(5), y=alt.value(5), text="text:N")
        )

        layers = [
            area_pred,
            *circles_train,
            line_true_interval,
            line_pred,
            line_true,
            text_metrics,
        ]
        if legend:
            # A legend of all of the marks, drawn by invisible points.
            scale = alt.Scale(domain=labels, range=list(colors.values()))
            layers.append(
                alt.Chart(pd.DataFrame({"label": labels}))
                .mark_point(filled=True, opacity=0)
                .encode(
                    color=alt.Color(
                        "label:N",
                        scale=scale,
                        legend=alt.Legend(symbolOpacity=1, rowPadding=5),
                        title=None,
                    ),
                    shape=alt.Shape(
                        "label:N",
                        scale=alt.Scale(domain=labels, range=shapes),
                        title=None,
                    ),
                )
                .add_params(click)
            )
        return alt.layer(*layers).properties(title=title, height=250, width=280)

    # Without weights is the same as with weights that assume the training data's share.
    chart1 = _plot_panel("Without Sample Weights", share_b_train)
    chart2 = _plot_panel("With Sample Weights", "share_b", legend=True)
    chart = (chart1 | chart2).add_params(share_val)
    return chart


chart = plot_sample_weights(df, df_true, df_train, df_metrics)
chart
