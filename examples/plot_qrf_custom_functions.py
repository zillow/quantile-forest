"""
Computing User-Specified Functions with QRFs
============================================

This example demonstrates how to extract the empirical distribution from a
quantile regression forest (QRF) to calculate user-specified functions of
interest. While a QRF is designed to estimate quantiles, their empirical
distributions can also be used to calculate other statistical quantities. In
this scenario, we compute the empirical cumulative distribution function
(ECDF) for test samples.
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_test_samples = 100


def predict(qrf, X, y_train, quantiles=0.5, what=None):
    """Custom prediction method that allows user-specified function.

    Parameters
    ----------
    qrf : BaseForestQuantileRegressor
        Quantile forest model object.
    X : array-like, of shape (n_samples, n_features)
        New test data.
    y_train : array-like, of shape (n_train_samples,) or (n_train_samples, n_outputs)
        Training target values, in the order used to fit the model.
    quantiles : float or list of floats, default=0.5
        Scalar or vector of quantiles for quantile prediction.
    what : Any, optional
        User-specified function for prediction used instead of quantiles.

    Returns
    -------
    Output array, or list of outputs of the user-specified function (if any).
    """
    if what is None:
        return qrf.predict(X, quantiles=quantiles)

    # Get the complete set of proximities for each sample: pairs of the index of
    # a training sample and the number of times that it is kept in the leaves
    # of the test sample, which is its weight for prediction (by default).
    proximities = qrf.proximity_counts(X)

    # For each sample, construct an array of the training responses used for prediction.
    y_train = np.asarray(y_train)
    s = [
        np.repeat(y_train[[i for i, _ in prox]], [c for _, c in prox], axis=0)
        for prox in proximities
    ]

    # Apply the user-specified function for each sample.
    return [what(s_i) for s_i in s]


# Load the Diabetes dataset.
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=n_test_samples, random_state=random_state
)

qrf = RandomForestQuantileRegressor(random_state=random_state)
qrf.fit(X_train, y_train)

# Define a user-specified function.
# Here we calculate the empirical cumulative distribution function (ECDF).
func = lambda x: sp.stats.ecdf(x).cdf

# Output list with the user-specified function applied to each sample's empirical distribution.
y_out = predict(qrf, X_test, y_train, what=func)

dfs = []
for idx in range(n_test_samples):
    # Get the quantiles (x-axis) and probabilities (y-axis) of the ECDF.
    quantiles = y_out[idx].quantiles.tolist()
    probabilities = y_out[idx].probabilities.tolist()

    df_i = pd.DataFrame(
        {
            "y_val": quantiles,
            "y_val2": quantiles[1:] + [np.nan],
            "proba": probabilities,
            "index": [idx] * len(quantiles),
        }
    )
    dfs.append(df_i)
df = pd.concat(dfs, ignore_index=True)


def plot_ecdf(df):
    """Plot the ECDF for test samples."""
    min_idx = df["index"].min()
    max_idx = df["index"].max()

    # Slider for determining the sample index for which the custom function is being visualized.
    slider = alt.binding_range(name="Test Sample Index: ", min=min_idx, max=max_idx, step=1)
    index_selection = alt.selection_point(
        value=0,
        bind=slider,
        empty=False,
        fields=["index"],
        on="click",
        nearest=True,
    )

    color = alt.condition(index_selection, alt.value("#006aff"), alt.value("lightgray"))
    opacity = alt.condition(index_selection, alt.value(1), alt.value(0.2))
    tooltip = [
        alt.Tooltip("index:Q", title="Sample Index"),
        alt.Tooltip("y_val:Q", title="Response Value"),
        alt.Tooltip("proba:Q", title="Probability"),
    ]

    circle = (
        alt.Chart(df)
        .mark_circle(color="#006aff", opacity=1, size=50)
        .encode(
            x=alt.X("y_val:Q", title="Response Value"),
            y=alt.Y("proba:Q", title="Probability"),
            color=color,
            opacity=opacity,
            tooltip=tooltip,
        )
    )

    rule = (
        alt.Chart(df)
        .mark_rule(color="#006aff", size=2)
        .encode(
            x=alt.X("y_val:Q", title="Response Value"),
            x2=alt.X2("y_val2:Q"),
            y=alt.Y("proba:Q", title="Probability"),
            color=color,
            opacity=opacity,
            tooltip=tooltip,
        )
    )

    # Ensure the selected sample index values overlay the unselected values.
    chart_base = circle + rule
    chart_selected = (circle + rule).transform_filter(index_selection)

    chart = (
        (chart_base + chart_selected)
        .add_params(index_selection)
        .properties(
            title="Empirical Cumulative Distribution Function (ECDF) Plot",
            height=400,
            width=650,
        )
    )
    return chart


chart = plot_ecdf(df)
chart
