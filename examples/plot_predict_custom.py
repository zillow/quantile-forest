"""
Computing User-Specified Functions with QRFs
============================================

An example that demonstrates a way of extracting the empirical distribution
from a quantile regression forest (QRF) for one or more samples in order to
calculate a user-specified function of interest. While a QRF is designed to
estimate quantiles from the empirical distribution calculated for each sample,
in many cases it may be useful to use the empirical distribution to calculate
other quantities of interest. Here, we calculate the ECDF for a test sample.
"""

from itertools import chain

import altair as alt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)

n_test_samples = 10


def predict(reg, X, quantiles=0.5, what=None):
    """Custom prediction method that allows user-specified function.

    Parameters
    ----------
    reg : BaseForestQuantileRegressor
        Quantile forest model object.
    X : array-like, of shape (n_samples, n_features)
        New test data.
    quantiles : float or list of floats, default=0.5
        Scalar or vector of quantiles for quantile prediction.
    what : Any, optional
        User-specified function for prediction used instead of quantiles.
        Must return a numeric vector.

    Returns
    -------
    Output array with the user-specified function applied (if any).
    """
    if what is None:
        return reg.predict(X, quantiles=quantiles)

    # Get the complete set of proximities (training indices) for each sample.
    proximities = reg.proximity_counts(X)

    # Retrieve the unsorted training responses from the model (stored in sorted order).
    reverse_sorter = np.argsort(reg.sorter_, axis=0)
    y_train = np.empty_like(reg.forest_.y_train).T
    for i in range(y_train.shape[1]):
        y_train[:, i] = np.asarray(reg.forest_.y_train)[i, reverse_sorter[:, i]]

    # For each sample, construct an array of the training responses used for prediction.
    s = [np.concatenate([np.repeat(y_train[i[0]], i[1]) for i in prox]) for prox in proximities]

    # Apply the user-specified function for each sample. Must return a numeric vector.
    y_out = np.array([what(s_i) for s_i in s])

    return y_out


X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test_samples, random_state=0)

reg = RandomForestQuantileRegressor().fit(X_train, y_train)

# Define a user-specified function; here we randomly sample 1000 values with replacement.
func = lambda x: np.random.choice(x, size=1000)

# Output array with the user-specified function applied to each sample's empirical distribution.
y_out = predict(reg, X_test, what=func)

dfs = []
for idx in range(n_test_samples):
    # Calculate the ECDF from output array.
    y_ecdf = [sp.stats.ecdf(y_i).cdf for y_i in y_out[idx].reshape(1, -1)]
    n_quantiles = len(list(chain.from_iterable([y_i.quantiles for y_i in y_ecdf])))

    df_i = pd.DataFrame(
        {
            "y_val": list(chain.from_iterable([y_i.quantiles for y_i in y_ecdf])),
            "y_val2": list(chain.from_iterable([y_i.quantiles for y_i in y_ecdf]))[1:] + [np.nan],
            "proba": list(chain.from_iterable([y_i.probabilities for y_i in y_ecdf])),
            "sample_idx": [idx] * n_quantiles,
        }
    )
    dfs.append(df_i)
df = pd.concat(dfs)


def plot_ecdf(df):
    min_idx = df["sample_idx"].min()
    max_idx = df["sample_idx"].max()
    slider = alt.binding_range(min=min_idx, max=max_idx, step=1, name="Sample Index:")
    sample_selection = alt.param(value=0, bind=slider, name="sample_idx")

    tooltip = [
        alt.Tooltip("y_val", title="Response Value"),
        alt.Tooltip("proba", title="Probability"),
    ]

    circles = (
        alt.Chart(df)
        .mark_circle(color="#006aff", opacity=1, size=50)
        .encode(
            x=alt.X("y_val", title="Response Value"),
            y=alt.Y("proba", title="Probability"),
            tooltip=tooltip,
        )
    )

    lines = (
        alt.Chart(df)
        .mark_line(color="#006aff", size=2)
        .encode(
            x=alt.X("y_val", title="Response Value"),
            x2=alt.X2("y_val2"),
            y=alt.Y("proba", title="Probability"),
            tooltip=tooltip,
        )
    )

    chart = (
        (circles + lines)
        .transform_filter(alt.datum.sample_idx == sample_selection)
        .add_params(sample_selection)
        .properties(
            height=400,
            width=650,
            title="Empirical Cumulative Distribution Function (ECDF) Plot",
        )
    )
    return chart


chart = plot_ecdf(df)
chart
