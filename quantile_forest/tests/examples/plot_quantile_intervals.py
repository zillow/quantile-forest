"""
Quantile Regression Forests Prediction Intervals
================================================

An example of how to use quantile regression forests to generate prediction
intervals on the California Housing dataset. Inspired by Figure 3 of
"Quantile Regression Forests" by Meinshausen:
https://jmlr.org/papers/v7/meinshausen06a.html.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

rng = check_random_state(0)

# Load the California Housing Prices dataset.
california = datasets.fetch_california_housing()
n_samples = min(california.target.size, 1000)
perm = rng.permutation(n_samples)
X = california.data[perm]
y = california.target[perm]

qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=0)

kf = KFold(n_splits=5)
kf.get_n_splits(X)

y_true = []
y_pred = []
y_pred_low = []
y_pred_upp = []

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = (
        X[train_index],
        X[test_index],
        y[train_index],
        y[test_index],
    )

    qrf.set_params(max_features=X_train.shape[1] // 3)
    qrf.fit(X_train, y_train)

    # Get predictions at 95% prediction intervals and median.
    y_pred_i = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])

    y_true.append(y_test)
    y_pred.append(y_pred_i[:, 1])
    y_pred_low.append(y_pred_i[:, 0])
    y_pred_upp.append(y_pred_i[:, 2])

df = pd.DataFrame(
    {
        "y_true": np.concatenate(y_true),
        "y_pred": np.concatenate(y_pred),
        "y_pred_low": np.concatenate(y_pred_low),
        "y_pred_upp": np.concatenate(y_pred_upp),
    }
).pipe(
    lambda x: x * 100_000  # convert to dollars
)


def plot_calibration_and_intervals(df):
    def plot_calibration(df):
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
                title="Fitted Values (Conditional Mean)",
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
            .mark_line(color="black", opacity=0.4, strokeDash=[2, 2])
            .encode(
                x=alt.X("var1:Q"),
                y=alt.Y("var2:Q"),
            )
        )

        chart = bar + tick_low + tick_upp + circle + diagonal
        return chart

    def plot_intervals(df):
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

        circle = base.mark_circle(size=30).encode(
            x=alt.X("idx:Q", axis=alt.Axis(format=",d"), title="Ordered Samples"),
            y=alt.Y(
                "y_true:Q",
                axis=alt.Axis(format="$,d"),
                title="Observed Values and Prediction Intervals",
            ),
            color=alt.value("#f2a619"),
            tooltip=tooltip,
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

        chart = bar + tick_low + tick_upp + circle
        return chart

    chart1 = plot_calibration(df).properties(height=250, width=325)
    chart2 = plot_intervals(df).properties(height=250, width=325)
    chart = chart1 | chart2

    return chart


chart = plot_calibration_and_intervals(df)
chart
