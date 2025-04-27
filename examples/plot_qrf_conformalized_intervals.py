"""
QRFs for Conformalized Quantile Regression
==========================================

This example demonstrates the use of a quantile regression forest (QRF) to
construct reliable prediction intervals using conformalized quantile
regression (CQR). While QRFs can estimate quantiles, they may require
additional calibration to provide reliable interval estimates. CQR provides
prediction intervals that attain valid coverage. In this example, we use CQR
to enhance QRF by producing prediction intervals that achieve a level of
coverage (i.e., the percentage of samples that actually fall within their
prediction interval) that is generally closer to the target level. This
example is adapted from `"Prediction intervals: Quantile Regression Forests"
<https://www.kaggle.com/code/carlmcbrideellis/prediction-intervals-quantile-regression-forests>`_
by Carl McBride Ellis.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_seed = 0
random_state = np.random.RandomState(random_seed)

n_samples = 900
coverages = np.linspace(0, 1, num=11, endpoint=True).round(1).tolist()  # the "coverage level"

strategies = {
    "qrf": "Quantile Regression Forest (QRF)",
    "cqr": "Conformalized Quantile Regression (CQR)",
}

# Load the California Housing dataset.
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
perm = random_state.permutation(min(len(X), n_samples))
X = X.iloc[perm]
y = y.iloc[perm]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)


def sort_y_values(y_test, y_pred, y_pis):
    """Sort the target values and predictions."""
    indices = np.argsort(y_test)
    return {
        "y_test": np.asarray(y_test)[indices],
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


def qrf_strategy(alpha, X_train, X_test, y_train, y_test, random_state=None):
    """QRF (baseline) strategy."""
    quantiles = [alpha / 2, 1 - alpha / 2]

    qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=random_state)
    qrf.fit(X_train, y_train)

    # Calculate the lower and upper quantile values on the test data.
    y_pred_interval = qrf.predict(X_test, quantiles=quantiles)
    y_pred_low = y_pred_interval[:, 0]
    y_pred_upp = y_pred_interval[:, 1]
    y_pis = np.stack([y_pred_low, y_pred_upp], axis=1)

    # Calculate the point predictions on the test data.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    y_values = sort_y_values(y_test, y_pred, y_pis)

    return pd.DataFrame(y_values).pipe(lambda x: x * 100_000).assign(strategy="qrf")


def cqr_strategy(alpha, X_train, X_test, y_train, y_test, random_state=None):
    """Conformalized Quantile Regression (CQR) strategy with a QRF."""
    quantiles = [alpha / 2, 1 - alpha / 2]

    # Create calibration set.
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.5, random_state=random_state
    )

    qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=random_state)
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
    s = np.quantile(conf_scores, np.clip((1 - alpha) * (1 + (1 / (len(y_calib)))), 0, 1))

    # Subtract `s` from the lower quantile and add it to the upper quantile.
    y_conf_low = y_pred_low - s
    y_conf_upp = y_pred_upp + s
    y_pis = np.stack([y_conf_low, y_conf_upp], axis=1)

    # Calculate the point predictions on the test data.
    y_pred = qrf.predict(X_test, quantiles="mean", aggregate_leaves_first=False)

    y_values = sort_y_values(y_test, y_pred, y_pis)

    return pd.DataFrame(y_values).pipe(lambda x: x * 100_000).assign(strategy="cqr")


# Get strategy outputs as a data frame.
dfs = []
for cov_frac in coverages:
    alpha = float(round(1 - cov_frac, 2))
    args = (alpha, X_train, X_test, y_train, y_test, random_seed)
    dfs.append(pd.concat([qrf_strategy(*args), cqr_strategy(*args)]).assign(alpha=alpha))
df = pd.concat(dfs)

metrics = (
    df.groupby(["alpha", "strategy"])
    .apply(
        lambda grp: pd.Series(
            {
                "coverage": coverage_score(grp["y_test"], grp["y_pred_low"], grp["y_pred_upp"]),
                "width": mean_width_score(grp["y_pred_low"], grp["y_pred_upp"]),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

# Merge the metrics into the data frame.
df = df.merge(metrics, on=["alpha", "strategy"], how="left")


def plot_prediction_intervals_by_strategy(df):
    """Plot prediction intervals by interval estimate strategy."""

    def _plot_prediction_intervals(df, domain):
        # Slider for varying the target coverage level.
        slider = alt.binding_range(name="Coverage Target: ", min=0, max=1, step=0.1)
        coverage_val = alt.param(name="coverage", value=0.9, bind=slider)

        click = alt.selection_point(bind="legend", fields=["y_label"])

        tooltip = [
            alt.Tooltip("y_test:Q", format="$,d", title="True Price"),
            alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price"),
            alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
            alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
            alt.Tooltip("y_label:N", title="Within Interval"),
        ]

        base = (
            alt.Chart(df)
            .transform_filter("round((1 - datum.alpha) * 100) / 100 == coverage")
            .transform_calculate(
                y_label=(
                    "((datum.y_test >= datum.y_pred_low) & (datum.y_test <= datum.y_pred_upp))"
                    " ? 'Yes' : 'No'"
                )
            )
        )

        bar = base.mark_bar(width=2).encode(
            x=alt.X("y_pred:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y=alt.Y("y_pred_low:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=alt.condition(click, alt.value("#e0f2ff"), alt.value("lightgray")),
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

        circle = (
            base.add_params(click)
            .mark_circle(size=30)
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
                color=alt.condition(
                    click,
                    alt.Color(
                        "y_label:N",
                        scale=alt.Scale(domain=["Yes", "No"], range=["#f2a619", "red"]),
                        title="Within Interval",
                    ),
                    alt.value("lightgray"),
                ),
                opacity=alt.condition(click, alt.value(1), alt.value(0)),
                tooltip=tooltip,
            )
        )

        diagonal = (
            alt.Chart(pd.DataFrame({"var1": domain, "var2": domain}))
            .mark_line(color="gray", opacity=0.4, strokeDash=[2, 2])
            .encode(
                x=alt.X("var1:Q"),
                y=alt.Y("var2:Q"),
            )
        )

        text_coverage = (
            base.transform_aggregate(
                alpha="mean(alpha)", coverage="mean(coverage)", groupby=["strategy"]
            )
            .transform_calculate(
                coverage_text=(
                    "'Coverage: ' + format(datum.coverage * 100, '.1f') + '%'"
                    " + ' (target = ' + format((1 - datum.alpha) * 100, '.1f') + '%)'"
                )
            )
            .mark_text(align="left", baseline="top", color="gray")
            .encode(
                x=alt.value(5),
                y=alt.value(5),
                text=alt.Text("coverage_text:N"),
            )
        )
        text_with = (
            base.transform_aggregate(width="mean(width)", groupby=["strategy"])
            .transform_calculate(width_text="'Interval Width: ' + format(datum.width, '$,d')")
            .mark_text(align="left", baseline="top", color="gray")
            .encode(
                x=alt.value(5),
                y=alt.value(20),
                text=alt.Text("width_text:N"),
            )
        )

        chart = (
            bar + tick_low + tick_upp + circle + diagonal + text_coverage + text_with
        ).add_params(coverage_val)

        return chart

    chart = alt.hconcat()
    for strategy in strategies.keys():
        domain = [
            int(np.min((df[["y_test", "y_pred"]].min(axis=0)))),  # min of all axes
            int(np.max((df[["y_test", "y_pred"]].max(axis=0)))),  # max of all axes
        ]
        df_i = df.query(f"strategy == '{strategy}'").reset_index(drop=True)
        base = _plot_prediction_intervals(df_i, domain)
        chart |= base.properties(height=225, width=300, title=strategies[strategy])

    return chart


chart = plot_prediction_intervals_by_strategy(df)
chart
