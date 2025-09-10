"""
Using Proximity Counts to Identify Similar Samples
==================================================

This example demonstrates the use of quantile regression forest (QRF)
proximity counts to identify similar samples. In this scenario, we train a QRF
on a noisy dataset to predict individual pixel values in an unsupervised
manner for denoising purposes; the target labels are not used during training.
We then retrieve the proximity values for the noisy test samples. We visualize
each test sample alongside a set of similar (non-noisy) training samples
determined by their proximity counts. These similar samples are ordered from
the highest to the lowest proximity count. This illustrates how proximity
counts can effectively identify similar samples even in noisy conditions.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_random_state

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_test_samples = 50
noise_std = 0.1

pixel_dim = (8, 8)  # pixel dimensions (width and height)
pixel_scale = 1000  # scale multiplier for combining clean and noisy values

# Load the Digits dataset.
X, y = datasets.load_digits(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=n_test_samples, random_state=random_state
)


def add_gaussian_noise(X, mean=0, std=0.1, random_state=None):
    """Add Gaussian noise to input data."""
    random_state = check_random_state(random_state)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    noise = random_state.normal(mean, std, X_scaled.shape)
    X_noisy = np.clip(X_scaled + noise, 0, 1)

    X_noisy = scaler.inverse_transform(X_noisy)

    if isinstance(X, pd.DataFrame):
        X_noisy = pd.DataFrame(X_noisy, index=X.index, columns=X.columns)

    return X_noisy


def combine_floats(df1, df2, scale=1000):
    """Combine two floats from separate data frames into a single number."""
    combined_df = df1 * scale + df2
    return combined_df


def extract_floats(combined_df, scale=1000):
    """Extract the original floats from the combined data frame."""
    df1 = np.floor(combined_df / scale)
    df2 = combined_df - (df1 * scale)
    return df1, df2


# Randomly add Gaussian noise to the training and test data.
X_train_noisy = X_train.pipe(add_gaussian_noise, std=noise_std, random_state=random_state)
X_test_noisy = X_test.pipe(add_gaussian_noise, std=noise_std, random_state=random_state)

# We set `max_samples_leaf=None` to ensure that every sample in the training
# data is stored in the leaf nodes. By doing this, we allow the model to
# consider all samples as potential candidates for proximity calculations.
qrf = RandomForestQuantileRegressor(
    n_estimators=250,
    max_features=1 / 3,
    max_samples_leaf=None,
    random_state=random_state,
)

# Fit the model to predict the non-noisy pixels from noisy pixels (i.e., to denoise).
# We fit a single multi-target model, with each pixel treated as a distinct target.
qrf.fit(X_train_noisy, X_train)

# Get the proximity counts.
proximities = qrf.proximity_counts(X_test_noisy)  # output is a list of tuples for each sample

df_prox = pd.DataFrame(
    {"prox": [[(i, *p) for i, p in enumerate(proximities[x])] for x in range(len(X_test))]}
)

df = (
    combine_floats(X_test, X_test_noisy, scale=pixel_scale)  # combine to reduce transmitted data
    .join(y_test)
    .reset_index()
    .join(df_prox)
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

# Create a data frame for looking up training proximities.
df_lookup = (
    combine_floats(X_train, X_train_noisy, scale=pixel_scale)  # combine to reduce transmitted data
    .join(y_train)
    .assign(**{"index": np.arange(len(X_train))})  # align with proximities, which are zero-indexed
)


def plot_digits_proximities(
    df,
    df_lookup,
    pixel_dim=(8, 8),
    pixel_scale=1000,
    n_prox=25,
    n_prox_per_row=5,
    subplot_spacing=10,
    height=225,
    width=225,
):
    """Plot Digits dataset proximities for test samples."""
    df = df[df["prox_idx"] < n_prox].copy()

    dim_x, dim_y = pixel_dim[0], pixel_dim[1]
    dgt_x, dgt_y = len(str(dim_x)), len(str(dim_y))

    pixel_cols = [f"pixel_{y:0{dgt_y}}_{x:0{dgt_x}}" for y in range(dim_y) for x in range(dim_x)]
    pixel_x = "split(datum.pixel, '_')[2]"
    pixel_y = "split(datum.pixel, '_')[1]"

    x_min = (df[pixel_cols] // pixel_scale).min().min()
    x_max = (df[pixel_cols] // pixel_scale).max().max()

    n_samples = df["index"].nunique()
    n_subplot_rows = n_prox // n_prox_per_row
    subplot_dim = (width - subplot_spacing * (n_subplot_rows - 1)) / n_subplot_rows

    # Slider for determining the test index for which the data is being visualized.
    slider = alt.binding_range(name="Test Sample Index: ", min=0, max=n_samples - 1, step=1)
    index_selection = alt.selection_point(value=0, bind=slider, fields=["index"])

    scale = alt.Scale(domain=[x_min, x_max], scheme="warmgreys")
    opacity = (alt.value(0), alt.value(0.5))

    base = alt.Chart(df).add_params(index_selection).transform_filter(index_selection)

    chart1 = (
        base.transform_filter("datum.prox_idx == 0")  # filter to one test sample row
        .transform_fold(fold=pixel_cols, as_=["pixel", "value"])
        .transform_calculate(value_clean=f"floor(datum.value / {pixel_scale})")
        .transform_calculate(value_noisy=f"datum.value - (datum.value_clean * {pixel_scale})")
        .transform_calculate(x=pixel_x, y=pixel_y)
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_noisy:Q", legend=None, scale=scale),
            opacity=alt.condition(alt.datum["value_noisy"] == 0, *opacity),
            tooltip=[
                alt.Tooltip("target:Q", title="Digit"),
                alt.Tooltip("value_noisy:Q", format=",.3f", title="Pixel Value"),
                alt.Tooltip("x:Q", title="Pixel X"),
                alt.Tooltip("y:Q", title="Pixel Y"),
            ],
        )
        .properties(height=height, width=width, title="Test Digit (noisy)")
    )

    chart2 = (
        base.transform_filter(f"datum.prox_idx < {n_prox}")  # filter to the display proximities
        .transform_lookup(
            lookup="prox_val",
            from_=alt.LookupData(df_lookup, key="index", fields=pixel_cols + ["target"]),
        )
        .transform_fold(fold=pixel_cols, as_=["pixel", "value"])
        .transform_calculate(value_clean=f"floor(datum.value / {pixel_scale})")
        .transform_calculate(x=pixel_x, y=pixel_y)
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_clean:Q", legend=None, scale=scale),
            opacity=alt.condition(alt.datum["value_clean"] == 0, *opacity),
            tooltip=[
                alt.Tooltip("prox_idx:N", title="Proximity Index"),
                alt.Tooltip("prox_cnt:Q", title="Proximity Count"),
                alt.Tooltip("target:Q", title="Digit"),
            ],
            facet=alt.Facet(
                "prox_idx:N",
                columns=n_prox // n_prox_per_row,
                title=None,
                header=alt.Header(labels=False, labelFontSize=0, labelPadding=0),
            ),
        )
        .properties(
            height=subplot_dim, width=subplot_dim, title=f"Proximity Digits (top {n_prox})"
        )
    )

    chart3 = (
        base.transform_filter("datum.prox_idx == 0")  # filter to one test sample row
        .transform_fold(fold=pixel_cols, as_=["pixel", "value"])
        .transform_calculate(value_clean=f"floor(datum.value / {pixel_scale})")
        .transform_calculate(x=pixel_x, y=pixel_y)
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_clean:Q", legend=None, scale=scale),
            opacity=alt.condition(alt.datum["value_clean"] == 0, *opacity),
            tooltip=[
                alt.Tooltip("target:Q", title="Digit"),
                alt.Tooltip("value_clean:Q", format=",.0f", title="Pixel Value"),
                alt.Tooltip("x:Q", title="Pixel X"),
                alt.Tooltip("y:Q", title="Pixel Y"),
            ],
        )
        .properties(title="Test Digit (original)", height=height, width=width)
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


chart = plot_digits_proximities(df, df_lookup, pixel_dim=pixel_dim, pixel_scale=pixel_scale)
chart
