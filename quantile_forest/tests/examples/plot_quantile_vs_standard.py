"""
Quantile Regression Forests vs. Random Forests
==============================================

An example comparison between a quantile regression forest and a standard
random forest regressor on a synthetic, right-skewed dataset. In a right-
skewed distribution, the mean is to the right of the median.
"""

import altair as alt
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

rng = check_random_state(0)

# Create right-skewed dataset.
n_samples = 5000
a, loc, scale = 5, -1, 1
skewnorm_rv = sp.stats.skewnorm(a, loc, scale)
skewnorm_rv.random_state = rng
y = skewnorm_rv.rvs(n_samples)
X = rng.randn(n_samples, 2) * y.reshape(-1, 1)

regr_rf = RandomForestRegressor(n_estimators=10, random_state=0)
regr_qrf = RandomForestQuantileRegressor(n_estimators=10, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

regr_rf.fit(X_train, y_train)
regr_qrf.fit(X_train, y_train)

y_pred_rf = regr_rf.predict(X_test)
y_pred_qrf = regr_qrf.predict(X_test, quantiles=0.5)

legend = {
    "Actual": "#c0c0c0",
    "RF (Mean)": "#f2a619",
    "QRF (Median)": "#006aff",
}

df = pd.DataFrame({"Actual": y_test, "RF (Mean)": y_pred_rf, "QRF (Median)": y_pred_qrf})


def plot_prediction_histograms(df, legend):
    click = alt.selection_point(fields=["estimator"], bind="legend")

    color = alt.condition(
        click,
        alt.Color("estimator:N", sort=list(legend.keys()), title=None),
        alt.value("lightgray"),
    )

    chart = (
        alt.Chart(df, width=alt.Step(6))
        .transform_fold(list(legend.keys()), as_=["estimator", "y_pred"])
        .transform_joinaggregate(total="count(*)", groupby=["estimator"])
        .transform_calculate(pct="1 / datum.total")
        .mark_bar()
        .encode(
            x=alt.X("estimator:N", axis=alt.Axis(labels=False, title=None)),
            y=alt.Y("sum(pct):Q", axis=alt.Axis(title="Frequency")),
            color=color,
            column=alt.Column(
                "y_pred:Q",
                bin=alt.Bin(maxbins=80),
                header=alt.Header(
                    labelExpr="datum.value % 1 == 0 ? floor(datum.value) : null",
                    labelOrient="bottom",
                    titleOrient="bottom",
                ),
                title="Actual and Predicted Target Values",
            ),
            tooltip=[
                alt.Tooltip("estimator:N", title=" "),
            ],
        )
        .add_params(click)
        .configure_facet(spacing=0)
        .configure_range(category=alt.RangeScheme(list(legend.values())))
        .configure_scale(bandPaddingInner=0.2)
        .configure_view(stroke=None)
    )

    return chart


chart = plot_prediction_histograms(df, legend)
chart
