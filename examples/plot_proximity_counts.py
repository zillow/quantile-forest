"""
Using Proximity Counts to Identify Similar Samples
==================================================

This example demonstrates the use of quantile regression forest (QRF)
proximity counts to identify similar samples in an unsupervised manner, as the
target values are not used during model fitting. In this scenario, we train a
QRF on a noisy dataset to predict individual pixel values (i.e., denoise). We
then retrieve the proximity values for samples in a noisy test set. For each
test sample digit, we visualize it alongside a set of similar (non-noisy)
training samples determined by their proximity counts, as well as the
non-noisy digit. The similar samples are ordered from the highest to the
lowest proximity count for each digit, arranged from left to right and top to
bottom. This example illustrates the effectiveness of proximity counts in
identifying similar samples, even when using noisy training and test data.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

rng = check_random_state(0)

n_test = 25
noise_std = 0.1

# Load the Digits dataset.
X, y = datasets.load_digits(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=0)


def add_gaussian_noise(X, mean=0, std=0.1, random_state=None):
    """Add Gaussian noise to input data."""
    if random_state is None:
        rng = check_random_state(0)
    else:
        rng = random_state

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    noise = rng.normal(mean, std, X_scaled.shape)
    X_noisy = X_scaled + noise

    X_noisy = scaler.inverse_transform(np.clip(X_noisy, 0, 1))

    if isinstance(X, pd.DataFrame):
        X_noisy = pd.DataFrame(X_noisy)
        X_noisy.index = X.index
        X_noisy.columns = X.columns

    return X_noisy


def combine_floats(df1, df2, scale=100):
    """Combine two floats from separate data frames into a single number."""
    combined_df = df1 * scale + df2
    return combined_df


def extract_floats(combined_df, scale=100):
    """Extract the original floats from the combined data frame."""
    df1 = np.floor(combined_df / scale)
    df2 = combined_df - (df1 * scale)
    return df1, df2


# Randomly add noise to the training and test data.
X_train_noisy = X_train.pipe(add_gaussian_noise, std=noise_std, random_state=rng)
X_test_noisy = X_test.pipe(add_gaussian_noise, std=noise_std, random_state=rng)

# We set `max_samples_leaf=None` to ensure that every sample in the training
# data is stored in the leaf nodes. By doing this, we allow the model to
# consider all samples as potential candidates for proximity calculations.
qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0)
qrf.fit(X_train_noisy, X_train)

# Get the proximity counts.
proximities = qrf.proximity_counts(X_test_noisy)

df_prox = pd.DataFrame(
    {"prox": [[(j, *p) for j, p in enumerate(proximities[i])] for i in range(len(X_test))]}
)

df = (
    combine_floats(X_test, X_test_noisy)
    .join(y_test)
    .reset_index()
    .join(df_prox)
    .iloc[:n_test]
    .explode("prox")
    .assign(
        **{
            "index": lambda x: pd.factorize(x["index"])[0],
            "prox_idx": lambda x: x["prox"].apply(lambda y: y[0]),
            "prox_val": lambda x: x["prox"].apply(lambda y: y[1]),
            "prox_cnt": lambda x: x["prox"].apply(lambda y: y[2]),
        }
    )
    .drop(columns=["prox"])
)

df_lookup = (
    combine_floats(X_train, X_train_noisy)
    .assign(**{"index": np.arange(len(X_train))})
    .join(y_train)
)


def plot_digits_proximities(
    df,
    df_lookup,
    n_prox=25,
    n_prox_per_row=5,
    subplot_spacing=10,
    height=225,
    width=225,
):
    n_samples = df["index"].nunique()
    n_subplot_rows = n_prox // n_prox_per_row
    subplot_dim = (width - subplot_spacing * (n_subplot_rows - 1)) / n_subplot_rows

    slider = alt.binding_range(
        min=0,
        max=n_samples - 1,
        step=1,
        name="Test Index: ",
    )

    idx_val = alt.selection_point(
        value=0,
        bind=slider,
        fields=["index"],
    )

    base = alt.Chart(df).add_params(idx_val).transform_filter(idx_val)

    chart1 = (
        base.transform_filter(f"datum.prox_idx == 0")
        .transform_fold(
            fold=[f"pixel_{y}_{x}" for y in range(8) for x in range(8)],
            as_=["pixel", "value"],
        )
        .transform_calculate(value_clean="floor(datum.value / 100)")
        .transform_calculate(value_noisy="datum.value - (datum.value_clean * 100)")
        .transform_calculate(x="substring(datum.pixel, 8, 9)", y="substring(datum.pixel, 6, 7)")
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_noisy:Q", legend=None, scale=alt.Scale(scheme="greys")),
            opacity=alt.condition(alt.datum["value_noisy"] == 0, alt.value(0), alt.value(0.67)),
            tooltip=[
                alt.Tooltip("target:Q", title="Digit"),
                alt.Tooltip("value_noisy:Q", format=".3f", title="Pixel Value"),
                alt.Tooltip("x:Q", title="Pixel X"),
                alt.Tooltip("y:Q", title="Pixel Y"),
            ],
        )
        .properties(height=height, width=width, title="Test Digit (noisy)")
    )

    chart2 = (
        base.mark_rect()
        .transform_filter(f"datum.prox_idx < {n_prox}")
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_clean:Q", legend=None, scale=alt.Scale(scheme="greys")),
            opacity=alt.condition(alt.datum["value_clean"] == 0, alt.value(0), alt.value(0.67)),
            tooltip=[
                alt.Tooltip("prox_cnt", title="Proximity Count"),
                alt.Tooltip("target:Q", title="Digit"),
            ],
            facet=alt.Facet(
                "prox_idx:N",
                columns=n_prox // n_prox_per_row,
                title=None,
                header=alt.Header(labels=False, labelFontSize=0, labelPadding=0),
            ),
        )
        .transform_lookup(
            lookup="prox_val",
            from_=alt.LookupData(
                df_lookup,
                key="index",
                fields=[f"pixel_{y}_{x}" for y in range(8) for x in range(8)] + ["target"],
            ),
        )
        .transform_fold(
            fold=[f"pixel_{y}_{x}" for y in range(8) for x in range(8)],
            as_=["pixel", "value"],
        )
        .transform_calculate(value_clean="floor(datum.value / 100)")
        .transform_calculate(x="substring(datum.pixel, 8, 9)", y="substring(datum.pixel, 6, 7)")
        .properties(
            height=subplot_dim, width=subplot_dim, title=f"Proximity Digits (top {n_prox})"
        )
    )

    chart3 = (
        base.transform_filter(f"datum.prox_idx == 0")
        .transform_fold(
            fold=[f"pixel_{y}_{x}" for y in range(8) for x in range(8)],
            as_=["pixel", "value"],
        )
        .transform_calculate(value_clean="floor(datum.value / 100)")
        .transform_calculate(x="substring(datum.pixel, 8, 9)", y="substring(datum.pixel, 6, 7)")
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_clean:Q", legend=None, scale=alt.Scale(scheme="greys")),
            opacity=alt.condition(alt.datum["value_clean"] == 0, alt.value(0), alt.value(0.67)),
            tooltip=[
                alt.Tooltip("target:Q", title="Digit"),
                alt.Tooltip("value_clean:Q", title="Pixel Value"),
                alt.Tooltip("x:Q", title="Pixel X"),
                alt.Tooltip("y:Q", title="Pixel Y"),
            ],
        )
        .properties(height=height, width=width, title="Test Digit (original)")
    )

    chart_spacer = alt.Chart(pd.DataFrame()).mark_rect().properties(width=subplot_dim * 2)

    chart = (
        (chart1 | chart_spacer | chart2 | chart_spacer | chart3)
        .configure_concat(spacing=0)
        .configure_facet(spacing=subplot_spacing)
        .configure_title(anchor="middle")
        .configure_view(strokeOpacity=0)
    )

    return chart


chart = plot_digits_proximities(df, df_lookup)
chart
