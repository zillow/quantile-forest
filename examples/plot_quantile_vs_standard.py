"""
Quantile Regression Forests vs. Random Forests
==============================================

This example compares the predictions generated by a quantile regression
forest (QRF) and a standard random forest regressor on a synthetic
right-skewed dataset. In a right-skewed distribution, the mean is to the right
of the median. As illustrated by a greater overlap in the frequencies of the
actual and predicted values, the median (quantile = 0.5) predicted by a
quantile regressor can be a more reliable estimator of a skewed distribution
than the mean.
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

alt.data_transformers.disable_max_rows()

random_seed = 0
rng = check_random_state(random_seed)

quantiles = np.linspace(0, 1, num=21, endpoint=True).round(2).tolist()

# Create right-skewed dataset.
n_samples = 5000
a, loc, scale = 5, -1, 1
skewnorm_rv = sp.stats.skewnorm(a, loc, scale)
skewnorm_rv.random_state = rng
y = skewnorm_rv.rvs(n_samples)
X = rng.randn(n_samples, 2) * y.reshape(-1, 1)

regr_rf = RandomForestRegressor(n_estimators=10, random_state=random_seed)
regr_qrf = RandomForestQuantileRegressor(n_estimators=10, random_state=random_seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

regr_rf.fit(X_train, y_train)
regr_qrf.fit(X_train, y_train)

y_pred_rf = regr_rf.predict(X_test)  # standard RF predictions (mean)
y_pred_qrf = regr_qrf.predict(X_test, quantiles=quantiles)  # QRF predictions (quantiles)

legend = {
    "Actual": "#c0c0c0",
    "RF (Mean)": "#f2a619",
    "QRF (Median)": "#006aff",
}

df = pd.concat(
    [
        pd.DataFrame({"actual": y_test, "rf": y_pred_rf, "qrf": y_pred_qrf[..., q_idx]}).assign(
            quantile=quantile
        )
        for q_idx, quantile in enumerate(quantiles)
    ],
    ignore_index=True,
)


def plot_prediction_histograms(df, legend):
    # Slider for varying the quantile value used for generating the QRF histogram.
    slider = alt.binding_range(
        min=0,
        max=1,
        step=0.5 if len(quantiles) == 1 else 1 / (len(quantiles) - 1),
        name="Predicted Quantile: ",
    )

    q_val = alt.selection_point(
        value=0.5,
        bind=slider,
        fields=["quantile"],
    )

    click = alt.selection_point(fields=["label"], bind="legend")

    color = alt.condition(
        click,
        alt.Color("label:N", sort=list(legend.keys()), title=None),
        alt.value("lightgray"),
    )

    chart = (
        alt.Chart(df)
        .add_params(q_val, click)
        .transform_filter(q_val)
        .transform_calculate(calculate=f"round(datum.actual * 10) / 10", as_="Actual")
        .transform_calculate(calculate=f"round(datum.rf * 10) / 10", as_="RF (Mean)")
        .transform_calculate(calculate=f"round(datum.qrf * 10) / 10", as_="QRF (Quantile)")
        .transform_fold(["Actual", "RF (Mean)", "QRF (Quantile)"], as_=["label", "value"])
        .mark_bar()
        .encode(
            x=alt.X(
                "value:N",
                axis=alt.Axis(
                    labelAngle=0,
                    labelExpr="datum.value % 0.5 == 0 ? datum.value : null",
                ),
                title="Actual and Predicted Target Values",
            ),
            y=alt.Y("count():Q", axis=alt.Axis(format=",d", title="Counts")),
            color=color,
            xOffset=alt.XOffset("label:N"),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("value:O", title="Value (binned)"),
                alt.Tooltip("count():Q", format=",d", title="Counts"),
            ],
        )
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .properties(
            height=400,
            width=650,
            title="Distribution of RF vs. QRF Predictions on Right-Skewed Distribution",
        )
    )
    return chart


chart = plot_prediction_histograms(df, legend)
chart
