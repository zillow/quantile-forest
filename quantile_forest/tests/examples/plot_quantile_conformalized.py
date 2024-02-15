"""
QRFs for Conformalized Quantile Regression
==========================================

An example that demonstrates the use of a quantile regression forest (QRF) to
construct reliable prediction intervals using conformalized quantile
regression (CQR). CQR offers prediction intervals that attain valid coverage,
while QRF may require additional calibration for reliable interval estimates.
Adapted from "Prediction intervals: Quantile Regression Forests" by Carl McBride
Ellis:
https://www.kaggle.com/code/carlmcbrideellis/prediction-intervals-quantile-regression-forests.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

strategies = {
    "qrf": "Quantile Regression Forest (QRF)",
    "cqr": "Conformalized Quantile Regression (CQR)",
}

random_state = 0
rng = check_random_state(random_state)

cov_pct = 95  # the "coverage level"
alpha = (100 - cov_pct) / 100

# Load the California Housing Prices dataset.
california = datasets.fetch_california_housing()
n_samples = min(california.target.size, 1000)
perm = rng.permutation(n_samples)
X = california.data[perm]
y = california.target[perm]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def sort_y_values(y_test, y_pred, y_pis):
    """Sort the target values and predictions."""
    indices = np.argsort(y_test)
    return {
        "y_test": y_test[indices],
        "y_pred": y_pred[indices],
        "y_pred_low": y_pis[:, 0][indices],
        "y_pred_upp": y_pis[:, 1][indices],
    }


def coverage_score(y_true, y_pred_low, y_pred_upp):
    """Effective coverage score obtained by the prediction intervals."""
    coverage = np.mean((y_pred_low <= y_true) & (y_pred_upp >= y_true))
    return float(coverage)


def mean_width_score(y_pred_low, y_pred_upp):
    """Effective mean width score obtained by the prediction intervals."""
    mean_width = np.abs(y_pred_upp - y_pred_low).mean()
    return float(mean_width)


def qrf_strategy(alpha, X_train, X_test, y_train, y_test):
    quantiles = [alpha / 2, 1 - alpha / 2]

    qrf = RandomForestQuantileRegressor(random_state=0)
    qrf.fit(X_train, y_train)

    # Calculate the lower and upper quantile values on the test data.
    y_pred_interval = qrf.predict(X_test, quantiles=quantiles)
    y_pred_low = y_pred_interval[:, 0]
    y_pred_upp = y_pred_interval[:, 1]
    y_pis = np.stack([y_pred_low, y_pred_upp], axis=1)

    # Calculate the point predictions on the test data.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    return sort_y_values(y_test, y_pred, y_pis)


def cqr_strategy(alpha, X_train, X_test, y_train, y_test):
    quantiles = [alpha / 2, 1 - alpha / 2]

    # Create calibration set.
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.5, random_state=0
    )

    qrf = RandomForestQuantileRegressor(random_state=0)
    qrf.fit(X_train, y_train)

    # Calculate the lower and upper quantile values on the test data.
    y_pred_interval = qrf.predict(X_test, quantiles=quantiles)
    y_pred_low = y_pred_interval[:, 0]
    y_pred_upp = y_pred_interval[:, 1]

    # Calculate the lower and upper quantile values on the calibration set.
    y_pred_interval_calib = qrf.predict(X_calib, quantiles=quantiles)
    y_pred_low_calib = y_pred_interval_calib[:, 0]
    y_pred_upp_calib = y_pred_interval_calib[:, 1]

    # Calculate the conformity scores on the calibration data.
    a = y_pred_low_calib - y_calib
    b = y_calib - y_pred_upp_calib
    conf_scores = (np.vstack((a, b)).T).max(axis=1)

    # Get the 1-alpha quantile `s` from the distribution of conformity scores.
    s = np.quantile(conf_scores, (1 - alpha) * (1 + (1 / (len(y_calib)))))

    # Subtract `s` from the lower quantile and add it to the upper quantile.
    y_conf_low = y_pred_low - s
    y_conf_upp = y_pred_upp + s
    y_pis = np.stack([y_conf_low, y_conf_upp], axis=1)

    # Calculate the point predictions on the test data.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    return sort_y_values(y_test, y_pred, y_pis)


# Get strategy outputs as a data frame.
args = (alpha, X_train, X_test, y_train, y_test)
df = pd.concat(
    [
        pd.DataFrame(qrf_strategy(*args)).pipe(lambda x: x * 100_000).assign(strategy="qrf"),
        pd.DataFrame(cqr_strategy(*args)).pipe(lambda x: x * 100_000).assign(strategy="cqr"),
    ]
)

# Add coverage and mean width metrics to the data frame.
df = df.merge(
    df.groupby("strategy")
    .apply(
        lambda x: pd.Series(
            {
                "coverage": coverage_score(x["y_test"], x["y_pred_low"], x["y_pred_upp"]),
                "width": mean_width_score(x["y_pred_low"], x["y_pred_upp"]),
            }
        )
    )
    .reset_index()
)


def plot_prediction_intervals(df, domain):
    click = alt.selection_point(fields=["y_label"], bind="legend")

    color_circle = alt.Color(
        "y_label:N",
        scale=alt.Scale(domain=["Yes", "No"], range=["#f2a619", "red"]),
        title="Within Interval",
    )
    color_bar = alt.value("#e0f2ff")

    tooltip = [
        alt.Tooltip("y_test:Q", format="$,d", title="True Price"),
        alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price"),
        alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
        alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
        alt.Tooltip("y_label:N", title="Within Interval"),
    ]

    base = alt.Chart(df).transform_calculate(
        y_label=(
            "((datum.y_test >= datum.y_pred_low) & (datum.y_test <= datum.y_pred_upp))"
            " ? 'Yes' : 'No'"
        )
    )

    circle = (
        base.mark_circle(size=30)
        .encode(
            x=alt.X(
                "y_pred:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="True Prices",
            ),
            y=alt.Y(
                "y_test:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="Predicted Prices",
            ),
            color=alt.condition(click, color_circle, alt.value("lightgray")),
            opacity=alt.condition(click, alt.value(1), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(click)
    )

    bar = base.mark_bar(width=2).encode(
        x=alt.X("y_pred:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
        y=alt.Y("y_pred_low:Q", scale=alt.Scale(clamp=True, domain=domain, padding=0), title=""),
        y2=alt.Y2("y_pred_upp:Q", title=None),
        color=alt.condition(click, color_bar, alt.value("lightgray")),
        opacity=alt.condition(click, alt.value(0.8), alt.value(0)),
        tooltip=tooltip,
    )

    tick = base.mark_tick(orient="horizontal", thickness=1, width=5).encode(
        x=alt.X("y_pred:Q", title=""),
        color=alt.value("#006aff"),
        opacity=alt.condition(click, alt.value(0.4), alt.value(0)),
    )
    tick_low = tick.encode(y=alt.Y("y_pred_low:Q", scale=alt.Scale(clamp=True), title=""))
    tick_upp = tick.encode(y=alt.Y("y_pred_upp:Q", scale=alt.Scale(clamp=True), title=""))

    diagonal = (
        alt.Chart(pd.DataFrame({"var1": domain, "var2": domain}))
        .mark_line(color="black", opacity=0.4, strokeDash=[2, 2])
        .encode(
            x=alt.X("var1:Q"),
            y=alt.Y("var2:Q"),
        )
    )

    text_coverage = (
        base.transform_aggregate(
            coverage="mean(coverage)", width="mean(width)", groupby=["strategy"]
        )
        .transform_calculate(
            coverage_text=(
                "'Coverage: ' + format(datum.coverage * 100, '.1f') + '%'"
                f" + ' (target = {cov_pct}%)'"
            )
        )
        .mark_text(align="left", baseline="top")
        .encode(
            x=alt.value(5),
            y=alt.value(5),
            text=alt.Text("coverage_text:N"),
        )
    )
    text_with = (
        base.transform_aggregate(
            coverage="mean(coverage)", width="mean(width)", groupby=["strategy"]
        )
        .transform_calculate(width_text="'Interval Width: ' + format(datum.width, '$,d')")
        .mark_text(align="left", baseline="top")
        .encode(
            x=alt.value(5),
            y=alt.value(20),
            text=alt.Text("width_text:N"),
        )
    )

    chart = bar + tick_low + tick_upp + circle + diagonal + text_coverage + text_with

    return chart


chart = alt.hconcat()
for strategy in strategies.keys():
    domain = [
        int(np.min((df[["y_test", "y_pred"]].min(axis=0)))),  # min of all axes
        int(np.max((df[["y_test", "y_pred"]].max(axis=0)))),  # max of all axes
    ]
    df_i = df.query(f"strategy == '{strategy}'").reset_index(drop=True)
    base = plot_prediction_intervals(df_i, domain)
    chart |= base.properties(height=225, width=300, title=strategies[strategy])
chart
