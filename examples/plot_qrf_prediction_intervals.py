"""
Quantile Regression Forests Prediction Intervals
================================================

This example demonstrates how to use quantile regression forests (QRF) to
generate prediction intervals on the California Housing dataset. The
visualization is inspired by Figure 3 of `"Quantile Regression Forests"
<https://jmlr.org/papers/v7/meinshausen06a.html>`_ by Meinshausen.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 1000

# Load the California Housing dataset.
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
perm = random_state.permutation(min(len(X), n_samples))
X = X.iloc[perm]
y = y.iloc[perm]

qrf = RandomForestQuantileRegressor(random_state=random_state)

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
kf.get_n_splits(X)

# Using k-fold cross-validation, get predictions for all samples.
data = {"y_true": [], "y_pred": [], "y_pred_low": [], "y_pred_upp": []}
for train_index, test_index in kf.split(X):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    qrf.set_params(max_features=X_train.shape[1] // 3)
    qrf.fit(X_train, y_train)

    # Get predictions at 95% prediction intervals and median.
    y_pred_i = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])

    data["y_true"].extend(y_test)
    data["y_pred"].extend(y_pred_i[:, 1])
    data["y_pred_low"].extend(y_pred_i[:, 0])
    data["y_pred_upp"].extend(y_pred_i[:, 2])

df = pd.DataFrame(data).pipe(lambda x: x * 100_000)  # convert to dollars


def plot_california_calibration_and_intervals(df):
    """Plot calibration and intervals on California Housing dataset."""

    def _plot_calibration(df):
        domain = [
            int(np.min(np.minimum(df["y_true"], df["y_pred"]))),  # min of both axes
            int(np.max(np.maximum(df["y_true"], df["y_pred"]))),  # max of both axes
        ]

        tooltip = [
            alt.Tooltip("y_true:Q", format="$,d", title="Actual Price"),
            alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price"),
            alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
            alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
        ]

        base = alt.Chart(df)

        circle = base.mark_circle(size=30).encode(
            x=alt.X(
                "y_pred:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="Fitted Values (conditional median)",
            ),
            y=alt.Y(
                "y_true:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="Observed Values",
            ),
            color=alt.value("#f2a619"),
            tooltip=tooltip,
        )

        bar = base.mark_bar(opacity=0.8, width=2).encode(
            x=alt.X("y_pred:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y=alt.Y("y_pred_low:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=alt.value("#e0f2ff"),
            tooltip=tooltip,
        )

        tick = base.mark_tick(opacity=0.4, orient="horizontal", thickness=1, width=5).encode(
            x=alt.X("y_pred:Q", title=""), color=alt.value("#006aff")
        )
        tick_low = tick.encode(y=alt.Y("y_pred_low:Q", title=""))
        tick_upp = tick.encode(y=alt.Y("y_pred_upp:Q", title=""))

        diagonal = (
            alt.Chart(
                pd.DataFrame({"var1": [domain[0], domain[1]], "var2": [domain[0], domain[1]]})
            )
            .mark_line(color="gray", opacity=0.4, strokeDash=[2, 2])
            .encode(
                x=alt.X("var1:Q"),
                y=alt.Y("var2:Q"),
            )
        )

        chart = bar + tick_low + tick_upp + circle + diagonal
        return chart

    def _plot_intervals(df):
        df = df.copy()

        # Order samples by interval width.
        y_pred_interval = df["y_pred_upp"] - df["y_pred_low"]
        sort_idx = np.argsort(y_pred_interval)
        df = df.iloc[sort_idx]
        df["idx"] = np.arange(len(df))

        # Center data, with the mean of the prediction interval at 0.
        mean = (df["y_pred_low"] + df["y_pred_upp"]) / 2
        df["y_true"] -= mean
        df["y_pred"] -= mean
        df["y_pred_low"] -= mean
        df["y_pred_upp"] -= mean

        x_domain = [0, len(df)]
        y_domain = [
            int(np.min(np.minimum(df["y_true"], df["y_pred"]))),  # min of both axes
            int(np.max(np.maximum(df["y_true"], df["y_pred"]))),  # max of both axes
        ]

        tooltip = [
            alt.Tooltip("idx:Q", format=",d", title="Sample Index"),
            alt.Tooltip("y_true:Q", format="$,d", title="Actual Price (Centered)"),
            alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price (Centered)"),
            alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
            alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
            alt.Tooltip("y_pred_width:Q", format="$,d", title="Prediction Interval Width"),
        ]

        base = alt.Chart(df).transform_calculate(
            y_pred_width=alt.datum["y_pred_upp"] - alt.datum["y_pred_low"]
        )

        bar = base.mark_bar(opacity=0.8, width=2).encode(
            x=alt.X("idx:Q", scale=alt.Scale(domain=x_domain, padding=0), title=""),
            y=alt.Y("y_pred_low:Q", scale=alt.Scale(domain=y_domain, padding=0), title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=alt.value("#e0f2ff"),
            tooltip=tooltip,
        )

        tick = base.mark_tick(opacity=0.4, orient="horizontal", thickness=1, width=5).encode(
            x=alt.X("idx:Q", title=""),
            color=alt.value("#006aff"),
        )
        tick_low = tick.encode(y=alt.Y("y_pred_low:Q", title=""))
        tick_upp = tick.encode(y=alt.Y("y_pred_upp:Q", title=""))

        circle = base.mark_circle(size=30).encode(
            x=alt.X("idx:Q", axis=alt.Axis(format=",d"), title="Ordered Samples"),
            y=alt.Y(
                "y_true:Q",
                axis=alt.Axis(format="$,d"),
                title="Observed Values and Prediction Intervals (centered)",
            ),
            color=alt.value("#f2a619"),
            tooltip=tooltip,
        )

        chart = bar + tick_low + tick_upp + circle
        return chart

    chart1 = _plot_calibration(df).properties(height=250, width=325)
    chart2 = _plot_intervals(df).properties(height=250, width=325)
    chart = chart1 | chart2

    return chart


chart = plot_california_calibration_and_intervals(df)
chart
