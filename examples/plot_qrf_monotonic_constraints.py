"""
Monotonic Constraints with Quantile Regression Forests
======================================================

This example demonstrates the use of monotonic constraints with a quantile
regression forest (QRF) on a toy dataset whose target increases with the first
of two features. A monotonic constraint (``monotonic_cst``) makes the
predictions increase with a feature, which here also makes the median more
accurate. But each tree is kept monotonic by clipping the target that each leaf
keeps to bounds derived from mean targets, which also clips much of the
variation that the quantiles rely on. As a result, the intervals of a
constrained QRF can be too narrow. To get both a monotonic median and reliable
intervals, we center the intervals of an unconstrained QRF on the median of a
constrained one and calibrate them with conformalized quantile regression (CQR)
on held-out data. The plot on the right shows them, along with the constrained
QRF's own intervals. Use the slider to set the target coverage: the constrained
QRF's intervals fall well short of every target, whereas the calibrated
intervals match it.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 5000
n_test_samples = 5000
coverages = np.linspace(0.5, 0.95, num=10).round(2).tolist()  # the target coverages


def get_true_median(x):
    """Get the true median of the toy dataset's target given the first feature."""
    return 2 * np.log1p(x)


def make_toy_dataset(n_samples, random_state=None):
    """Make a toy dataset whose target increases with the first feature."""
    random_state = check_random_state(random_state)
    X = random_state.uniform(0, 10, size=(n_samples, 2))  # the second feature is noise
    noise = (0.25 + 0.1 * X[:, 0]) * random_state.normal(size=n_samples)
    y = get_true_median(X[:, 0]) + noise
    return X, y


X, y = make_toy_dataset(n_samples, random_state=random_state)
X_test, y_test = make_toy_dataset(n_test_samples, random_state=random_state)

# Hold out data for calibrating the intervals.
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

qrf = RandomForestQuantileRegressor(min_samples_leaf=10, random_state=random_state)
qrf.fit(X_train, y_train)

# Constrain the predictions to increase with the first feature.
qrf_mono = RandomForestQuantileRegressor(
    min_samples_leaf=10, monotonic_cst=[1, 0], random_state=random_state
)
qrf_mono.fit(X_train, y_train)


def get_quantiles(coverage):
    """Get the quantiles of the lower bound, median, and upper bound of an interval."""
    return np.round([(1 - coverage) / 2, 0.5, (1 + coverage) / 2], 3).tolist()


def predict_centered(X, coverage):
    """Predict the unconstrained QRF intervals centered on the constrained median."""
    y_pred = qrf.predict(X, quantiles=get_quantiles(coverage))
    y_pred_mono = qrf_mono.predict(X, quantiles=0.5)
    return y_pred - y_pred[:, [1]] + y_pred_mono[:, np.newaxis]


def get_scale(coverage):
    """Get the CQR adjustment of the centered intervals, relative to their half-widths."""
    # The conformity scores are scaled by the half-width of each interval, so that the
    # intervals keep their shape.
    y_pred_calib = predict_centered(X_calib, coverage)
    half_width = (y_pred_calib[:, 2] - y_pred_calib[:, 0]) / 2
    scores = np.maximum(y_pred_calib[:, 0] - y_calib, y_calib - y_pred_calib[:, 2]) / half_width
    n_calib = len(scores)
    quantile = min(1, np.ceil((n_calib + 1) * coverage) / n_calib)
    return np.quantile(scores, quantile, method="higher")


# Calibrate the centered intervals with CQR on the held-out data, for each target coverage.
scales = {coverage: get_scale(coverage) for coverage in coverages}


def predict_calibrated(X, coverage):
    """Predict the centered intervals, calibrated with CQR."""
    y_pred = predict_centered(X, coverage)
    half_width = (y_pred[:, 2] - y_pred[:, 0]) / 2
    y_pred[:, 0] -= scales[coverage] * half_width
    y_pred[:, 2] += scales[coverage] * half_width
    return y_pred


# Predict along the first feature, with the second feature fixed.
X_grid = np.column_stack([np.linspace(0, 10, 201), np.full(201, 5.0)])

predictors = {
    "unconstrained": lambda X, coverage: qrf.predict(X, quantiles=get_quantiles(coverage)),
    "constrained": lambda X, coverage: qrf_mono.predict(X, quantiles=get_quantiles(coverage)),
    "calibrated": predict_calibrated,
}

df = pd.DataFrame({"x": X_grid[:, 0], "y_true": get_true_median(X_grid[:, 0])})
dfs_intervals, metrics = [], {}
for coverage in coverages:
    df_intervals = pd.DataFrame({"coverage": coverage, "x": X_grid[:, 0]})
    metrics[coverage] = {}
    for name, predict in predictors.items():
        y_pred_grid = predict(X_grid, coverage)
        df[f"{name}_median"] = y_pred_grid[:, 1]  # the same for every target coverage
        df_intervals[f"{name}_low"], df_intervals[f"{name}_upp"] = y_pred_grid[:, [0, 2]].T

        # Evaluate the predictions on the test data.
        y_pred_test = predict(X_test, coverage)
        metrics[coverage][name] = {
            "coverage": np.mean((y_pred_test[:, 0] <= y_test) & (y_test <= y_pred_test[:, 2])),
            "width": np.mean(y_pred_test[:, 2] - y_pred_test[:, 0]),
            "median_error": np.mean(np.abs(y_pred_test[:, 1] - get_true_median(X_test[:, 0]))),
        }
    dfs_intervals.append(df_intervals)
df_intervals = pd.concat(dfs_intervals, ignore_index=True)

# A sample of the test data, for display.
df_test = pd.DataFrame({"x": X_test[:200, 0], "y": y_test[:200]})


def plot_monotonic_constraints(df, df_intervals, df_test, metrics):
    """Plot the predictions of the unconstrained and constrained QRFs."""
    colors = {
        "Test Observations": "#f2a619",
        "True Median": "gray",
        "Predicted Median": "#006aff",
        "Predicted Interval": "#e0f2ff",
        "Constrained QRF's Interval": "#99c2ff",
    }
    labels = list(colors.keys())
    # The legend symbols. The lines are drawn as thin bars rather than strokes, as
    # stroked symbols would space the legend entries unevenly.
    bar = "M-1,-0.2L1,-0.2L1,0.2L-1,0.2Z"
    shapes = ["circle", bar, bar, "circle", "circle"]

    # Slider for varying the target coverage of the prediction intervals.
    slider = alt.binding_range(
        name="Coverage Target: ", min=coverages[0], max=coverages[-1], step=0.05
    )
    coverage_val = alt.param(name="coverage", value=0.8, bind=slider)

    filter_coverage = "round(datum.coverage * 100) == round(coverage * 100)"

    # Clicking an entry of the legend highlights its marks (shift-click for more).
    click = alt.selection_point(name="click", fields=["label"], bind="legend")

    def _opacity(value):
        return alt.condition(click, alt.value(value), alt.value(0.1))

    x = alt.X("x:Q", scale=alt.Scale(nice=False), title="X (First Feature)")
    y_scale = alt.Scale(domain=[-1, 10])  # with room for the widest intervals and the metrics

    def _plot_panel(title, median, interval, inner_interval=None, legend=False):
        tooltip = [
            alt.Tooltip("x:Q", format=",.3f", title="X"),
            alt.Tooltip("y_true:Q", format=",.3f", title="True Median"),
            alt.Tooltip(f"{median}:Q", format=",.3f", title="Predicted Median"),
        ]

        intervals = [(interval, labels[3])]
        if inner_interval is not None:
            intervals.append((inner_interval, labels[4]))

        areas = [
            alt.Chart(df_intervals.assign(label=label))
            .transform_filter(filter_coverage)
            .mark_area(color=colors[label])
            .encode(
                x=x,
                y=alt.Y(f"{name}_low:Q", scale=y_scale, title="Y"),
                y2=f"{name}_upp:Q",
                opacity=_opacity(0.8),
                tooltip=[
                    alt.Tooltip("x:Q", format=",.3f", title="X"),
                    alt.Tooltip(f"{name}_low:Q", format=",.3f", title="Predicted Lower Y"),
                    alt.Tooltip(f"{name}_upp:Q", format=",.3f", title="Predicted Upper Y"),
                ],
            )
            for name, label in intervals
        ]

        circle_test = (
            alt.Chart(df_test.assign(label=labels[0]))
            .mark_circle(color=colors[labels[0]])
            .encode(
                x=x,
                y="y:Q",
                opacity=_opacity(0.7),
                tooltip=[
                    alt.Tooltip("x:Q", format=",.3f", title="X"),
                    alt.Tooltip("y:Q", format=",.3f", title="Y"),
                ],
            )
        )

        line_pred = (
            alt.Chart(df.assign(label=labels[2]))
            .mark_line(color=colors[labels[2]], size=4)
            .encode(x=x, y=f"{median}:Q", opacity=_opacity(1), tooltip=tooltip)
        )

        # Draw the true median over the predicted median, where they overlap.
        line_true = (
            alt.Chart(df.assign(label=labels[1]))
            .mark_line(color=colors[labels[1]], size=2)
            .encode(x=x, y="y_true:Q", opacity=_opacity(1), tooltip=tooltip)
        )

        # The metrics of the panel's interval at the target coverage.
        text_metrics = (
            alt.Chart(df_text)
            .transform_filter(filter_coverage)
            .mark_text(align="left", baseline="top", color="gray", lineBreak="\n")
            .encode(x=alt.value(5), y=alt.value(5), text=f"{interval}:N")
        )

        layers = areas + [circle_test, line_pred, line_true, text_metrics]
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
                        "label:N", scale=alt.Scale(domain=labels, range=shapes), title=None
                    ),
                )
                .add_params(click)
            )
        return alt.layer(*layers).properties(title=title, height=250, width=280)

    def _text(metrics, coverage, coverage_label="Coverage", constrained_coverage=None):
        lines = [
            f"Median Error: {metrics['median_error']:.3f}",
            f"Interval Width: {metrics['width']:.2f}",
        ]
        if constrained_coverage is not None:
            lines.append(f"Constrained QRF's Coverage: {constrained_coverage:.1%}")
        lines.append(f"{coverage_label}: {metrics['coverage']:.1%} (target = {coverage:.0%})")
        return "\n".join(lines)

    # The text of each panel's metrics, at each target coverage.
    df_text = pd.DataFrame(
        {
            "coverage": coverages,
            "unconstrained": [_text(metrics[c]["unconstrained"], c) for c in coverages],
            "calibrated": [
                _text(
                    metrics[c]["calibrated"],
                    c,
                    coverage_label="Calibrated Coverage",
                    constrained_coverage=metrics[c]["constrained"]["coverage"],
                )
                for c in coverages
            ],
        }
    )

    chart1 = _plot_panel("Unconstrained QRF", "unconstrained_median", "unconstrained")
    chart2 = _plot_panel(
        "Constrained Median, Calibrated Interval",
        "constrained_median",
        "calibrated",
        inner_interval="constrained",
        legend=True,
    )
    chart = (chart1 | chart2).add_params(coverage_val)
    return chart


chart = plot_monotonic_constraints(df, df_intervals, df_test, metrics)
chart
