"""
Weighted vs. Unweighted Quantile Estimates
==========================================

An example comparison of the prediction runtime when using a quantile
regression forest with weighted and unweighted quantiles to compute the
predicted output values. A standard random forest regressor is included for
comparison.
"""

import time
from contextlib import contextmanager

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor


@contextmanager
def timing():
    t0 = time.time()
    yield lambda: (t1 - t0)
    t1 = time.time()


X, y = datasets.make_regression(n_samples=250, n_features=4, n_targets=5, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

legend = {
    "RF": "#f2a619",
    "QRF Weighted Quantile": "#006aff",
    "QRF Unweighted Quantile": "#001751",
}

est_sizes = [1, 5, 10, 25, 50, 100]
n_repeats = 10

timings = np.empty((len(est_sizes), n_repeats, 3))
for i, n_estimators in enumerate(est_sizes):
    for j in range(n_repeats):
        result = {}

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=0,
        )
        qrf = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            max_samples_leaf=None,
            random_state=0,
        )

        rf.fit(X_train, y_train)
        qrf.fit(X_train, y_train)

        with timing() as rf_time:
            _ = rf.predict(X_test)
        with timing() as qrf_weighted_time:
            _ = qrf.predict(X_test, quantiles=0.5, weighted_quantile=True)
        with timing() as qrf_unweighted_time:
            _ = qrf.predict(X_test, quantiles=0.5, weighted_quantile=False)

        timings[i, j, :] = [rf_time(), qrf_weighted_time(), qrf_unweighted_time()]
        timings[i, j, :] *= 1000  # convert from milliseconds to seconds

timings /= timings.min()  # normalize by minimum runtime
timings = np.transpose(timings, axes=[2, 0, 1])  # put the estimator name first

data = {"name": [], "n_estimators": [], "iteration": [], "runtime": []}
for i, name in enumerate(legend):
    for j in range(timings.shape[1]):
        data["name"].extend([name] * n_repeats)
        data["n_estimators"].extend([est_sizes[j]] * n_repeats)
        data["iteration"].extend(list(range(n_repeats)))
        data["runtime"].extend(timings[i, j])

df = (
    pd.DataFrame(data)
    .groupby(["name", "n_estimators"])
    .agg({"runtime": ["mean", "std"]})
    .pipe(lambda x: x.set_axis(["_".join(map(str, col)) for col in x.columns], axis=1))
    .reset_index()
    .assign(
        **{
            "mean": lambda x: x["runtime_mean"],
            "std": lambda x: x["runtime_std"],
            "ymin": lambda x: x["mean"] - (x["std"] / 2),
            "ymax": lambda x: x["mean"] + (x["std"] / 2),
        }
    )
)


def plot_timings_by_size(df, legend):
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

    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("n_estimators:Q", title="Number of Estimators"),
            y=alt.Y("mean:Q", title="Prediction Runtime (normalized)"),
            color=color,
        )
    )

    area = (
        alt.Chart(df)
        .mark_area(opacity=0.1)
        .encode(
            x=alt.X("n_estimators:Q"),
            y=alt.Y("ymin:Q"),
            y2=alt.Y2("ymax:Q"),
            color=color,
            tooltip=[
                alt.Tooltip("name:N", title="Estimator Name"),
                alt.Tooltip("n_estimators:Q", format=",d", title="Number of Estimators"),
                alt.Tooltip("mean:Q", format=",.3f", title="Average Runtime"),
                alt.Tooltip("ymin:Q", format=",.3f", title="Minimum Runtime"),
                alt.Tooltip("ymax:Q", format=",.3f", title="Maximum Runtime"),
            ],
        )
    )

    chart = (
        (line + area)
        .add_params(click)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .properties(height=400, width=650)
    )

    return chart


chart = plot_timings_by_size(df, legend)
chart
