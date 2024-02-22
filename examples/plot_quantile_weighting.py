"""
Weighted vs. Unweighted Quantile Runtimes
=========================================

An example comparison of the prediction runtime when using a quantile
regression forest with weighted and unweighted quantiles to compute the
predicted output values. While weighted and unweighted quantiles produce
identical outputs, the relative runtime of the methods depends on the number
of training samples and the total number of leaf samples across all trees used
to calculate the quantiles. A standard random forest regressor is included for
comparison.
"""

import time
from contextlib import contextmanager

import altair as alt
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

from quantile_forest import RandomForestQuantileRegressor

samples = [100, 175, 250, 325, 500]
estimators = [10, 25, 50, 75, 100]
repeats = 3


@contextmanager
def timing():
    t0 = time.process_time()
    yield lambda: (t1 - t0)
    t1 = time.process_time()


legend = {
    "RF": "#f2a619",
    "QRF Weighted Quantile": "#006aff",
    "QRF Unweighted Quantile": "#001751",
}

dataset = datasets.make_regression(
    n_samples=max(samples), n_features=3, n_targets=1, random_state=0
)

# Populate data with timing results over samples and estimators.
data = {"name": [], "n_samples": [], "n_estimators": [], "iteration": [], "runtime": []}
for n_samples in samples:
    X = dataset[0][:n_samples, :]
    y = dataset[1][:n_samples]
    for n_estimators in estimators:
        for repeat in range(repeats):
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=7,
                random_state=0,
            )
            qrf = RandomForestQuantileRegressor(
                n_estimators=n_estimators,
                max_depth=7,
                max_samples_leaf=None,
                random_state=0,
            )

            rf.fit(X, y)
            qrf.fit(X, y)

            with timing() as rf_time:
                _ = rf.predict(X)
            with timing() as qrf_weighted_time:
                _ = qrf.predict(X, quantiles=0.5, weighted_quantile=True)
            with timing() as qrf_unweighted_time:
                _ = qrf.predict(X, quantiles=0.5, weighted_quantile=False)

            timings = [rf_time(), qrf_weighted_time(), qrf_unweighted_time()]

            for name, runtime in zip(legend.keys(), timings):
                runtime *= 1000  # convert from milliseconds to seconds

                data["name"].extend([name])
                data["n_samples"].extend([n_samples])
                data["n_estimators"].extend([n_estimators])
                data["iteration"].extend([repeat])
                data["runtime"].extend([runtime])

df = (
    pd.DataFrame(data)
    .groupby(["name", "n_samples", "n_estimators"])
    .agg({"runtime": ["mean", "std"]})
    .pipe(lambda x: x.set_axis(["_".join(map(str, col)) for col in x.columns], axis=1))
    .reset_index()
    .assign(
        **{
            "ymean": lambda x: x["runtime_mean"],
            "ystd": lambda x: x["runtime_std"],
            "ymin": lambda x: x["ymean"] - (x["ystd"] / 2),
            "ymax": lambda x: x["ymean"] + (x["ystd"] / 2),
        }
    )
    .drop(columns=["runtime_mean", "runtime_std"])
)


def plot_timings_by_factor(df, legend, constant, factor, factor_title):
    max_value = df[constant].max()

    click = alt.selection_point(fields=["name"], bind="legend")

    color = alt.condition(
        click,
        alt.Color(
            "name:N",
            legend=alt.Legend(symbolOpacity=1),
            sort=list(legend.keys()),
            title=None,
        ),
        alt.value("lightgray"),
    )

    base = (
        alt.Chart(df)
        .transform_filter(alt.datum[constant] == max_value)  # hold this factor constant
        .transform_joinaggregate(min_runtime="min(ymin):Q", groupby=[constant])
        .transform_calculate(ymean_norm="datum.ymean / datum.min_runtime")
        .transform_calculate(ystd_norm="datum.ystd / datum.min_runtime")
        .transform_calculate(ymin_norm="datum.ymin / datum.min_runtime")
        .transform_calculate(ymax_norm="datum.ymax / datum.min_runtime")
    )

    line = base.mark_line().encode(
        x=alt.X(f"{factor}:Q", scale=alt.Scale(nice=False), title=factor_title),
        y=alt.Y(
            "ymean_norm:Q",
            scale=alt.Scale(zero=False),
            title="Prediction Runtime (normalized)",
        ),
        color=color,
    )

    area = base.mark_area(opacity=0.1).encode(
        x=alt.X(f"{factor}:Q"),
        y=alt.Y("ymin_norm:Q"),
        y2=alt.Y2("ymax_norm:Q"),
        color=color,
        tooltip=[
            alt.Tooltip("name:N", title="Estimator Name"),
            alt.Tooltip(f"{factor}:Q", format=",d", title=factor_title),
            alt.Tooltip("ymean_norm:Q", format=",.3f", title="Average Runtime"),
            alt.Tooltip("ymin_norm:Q", format=",.3f", title="Minimum Runtime"),
            alt.Tooltip("ymax_norm:Q", format=",.3f", title="Maximum Runtime"),
        ],
    )

    text = (
        base.transform_aggregate(coverage="mean(constant)", groupby=[constant])
        .transform_calculate(text=(f"'{constant} = ' + format({alt.datum[constant]}, ',d')"))
        .mark_text(align="left", baseline="top")
        .encode(
            x=alt.value(5),
            y=alt.value(5),
            text=alt.Text("text:N"),
        )
    )

    chart = (
        (line + area + text)
        .add_params(click)
        .properties(height=200, width=300, title=f"Runtime by {factor_title}")
    )

    return chart


chart1 = plot_timings_by_factor(df, legend, "n_samples", "n_estimators", "Number of Estimators")
chart2 = plot_timings_by_factor(df, legend, "n_estimators", "n_samples", "Number of Samples")
chart = (chart1 | chart2).configure_range(category=alt.RangeScheme(list(legend.values())))
chart
