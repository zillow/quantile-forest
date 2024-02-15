"""
Predicting with Quantile Regression Forests
===========================================

An example that demonstrates the use of a quantile regression forest to
predict a conditional median and prediction intervals. The example compares
the predictions to a ground truth function used to generate noisy samples.
"""

import altair as alt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

n_samples = 1000
bounds = [0, 10]


def make_toy_dataset(n_samples, bounds, random_seed=0):
    rng = np.random.RandomState(random_seed)

    x = rng.uniform(*bounds, size=n_samples)
    f = x * np.sin(x)

    sigma = 0.25 + x / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = f + noise

    return np.atleast_2d(x).T, y


X, y = make_toy_dataset(n_samples, bounds)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_sampled = np.atleast_2d(np.linspace(*bounds, n_samples)).T
y_sampled = (X_sampled * np.sin(X_sampled)).reshape(-1)

qrf = RandomForestQuantileRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
qrf.fit(X_train, y_train)

y_pred = qrf.predict(X_sampled, quantiles=[0.025, 0.5, 0.975])

df_train = pd.DataFrame(
    {
        "X_sampled": X_sampled.reshape(-1),
        "y_sampled": y_sampled,
        "y_pred_low": y_pred[:, 0],
        "y_pred_med": y_pred[:, 1],
        "y_pred_upp": y_pred[:, 2],
    }
)

df_test = pd.DataFrame(
    {
        "X_test": X_test.reshape(-1),
        "y_test": y_test,
    }
)


def plot_fit_and_intervals(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train = df_train.assign(
        **{
            "y_true_label": "f(x) = x sin(x)",
            "y_pred_label": "Predicted Median",
            "y_area_label": "Predicted 95% Interval",
        }
    )

    df_test["point_label"] = "Test Observations"

    points = (
        alt.Chart(df_test)
        .mark_circle(color="#f2a619")
        .encode(
            x=alt.X("X_test:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y_test:Q", title=""),
            color=alt.Color("point_label:N", scale=alt.Scale(range=["#f2a619"]), title=None),
            tooltip=[
                alt.Tooltip("X_test:Q", format=",.3f", title="X"),
                alt.Tooltip("y_test:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_true = (
        alt.Chart(df_train)
        .mark_line(color="black", size=3)
        .encode(
            x=alt.X("X_sampled:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y_sampled:Q", title="f(x)"),
            color=alt.Color("y_true_label:N", scale=alt.Scale(range=["black"]), title=None),
            tooltip=[
                alt.Tooltip("X_sampled:Q", format=",.3f", title="X"),
                alt.Tooltip("y_sampled:Q", format=",.3f", title="Y"),
            ],
        )
    )

    line_pred = (
        alt.Chart(df_train)
        .mark_line(color="#006aff", size=5)
        .encode(
            x=alt.X("X_sampled:Q", scale=alt.Scale(nice=False)),
            y=alt.Y("y_pred_med:Q", title=""),
            color=alt.Color("y_pred_label:N", scale=alt.Scale(range=["#006aff"]), title=None),
            tooltip=[
                alt.Tooltip("X_sampled:Q", format=",.3f", title="X"),
                alt.Tooltip("y_sampled:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred_med:Q", format=",.3f", title="Predicted Y"),
            ],
        )
    )

    area_pred = (
        alt.Chart(df_train)
        .mark_area(color="#e0f2ff", opacity=0.8)
        .encode(
            x=alt.X("X_sampled:Q", scale=alt.Scale(nice=False), title="x"),
            y=alt.Y("y_pred_low:Q", title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            tooltip=[
                alt.Tooltip("X_sampled:Q", format=",.3f", title="X"),
                alt.Tooltip("y_sampled:Q", format=",.3f", title="Y"),
                alt.Tooltip("y_pred_med:Q", format=",.3f", title="Predicted Y"),
                alt.Tooltip("y_pred_low:Q", format=",.3f", title="Predicted Lower Y"),
                alt.Tooltip("y_pred_upp:Q", format=",.3f", title="Predicted Upper Y"),
            ],
        )
    )

    # For desired legend ordering.
    blank = (
        alt.Chart(pd.DataFrame({"y_area_label": ["Predicted 95% Interval"]}))
        .mark_area(color="#e0f2ff")
        .encode(
            color=alt.Color("y_area_label:N", scale=alt.Scale(range=["#e0f2ff"]), title=None),
        )
    )

    chart = (
        (area_pred + points + line_true + line_pred + blank)
        .resolve_scale(color="independent")
        .properties(height=400, width=650)
    )

    return chart


chart = plot_fit_and_intervals(df_train, df_test)
chart
