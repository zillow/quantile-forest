"""
Using Proximity Counts to Identify Similar Samples
==================================================

This example demonstrates the use of quantile regression forest (QRF)
proximity counts to identify similar samples. In this scenario, we train a QRF
to predict individual pixel values on a corrupted dataset. We then retrieve
the proximity values for samples in a corrupted test set. For each test sample
digit, we visualize it alongside a set of similar (uncorrupted) training
samples determined by their proximity counts, as well as the uncorrupted
digit. The similar samples are ordered from the highest to the lowest
proximity count for each digit, arranged from left to right and top to bottom.
This example illustrates the effectiveness of proximity counts in identifying
similar samples, even when using noisy training and test data.
"""

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

alt.data_transformers.disable_max_rows()

n_test_samples = 10
corrupt_frac = 0.5

# Load the Digits dataset.
X, y = datasets.load_digits(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test_samples, random_state=0)


def randomly_mask_values(df, n=None, frac=None, seed=0):
    """Randomly mask a fraction of the values in a data frame with NaN."""
    np.random.seed(seed)

    df = df.copy()

    if n is not None:
        num_nan = n
    elif frac is not None:
        num_nan = int(df.size * frac)

    random_rows = np.random.randint(0, df.shape[0], num_nan)
    random_cols = np.random.randint(0, df.shape[1], num_nan)

    df.values[random_rows, random_cols] = np.nan

    return df


def digits_wide_to_long(df):
    """Transform the input wide data frame to long format.

    Each row of the input data frame represents one digit sample, with each
    column representing a pixel value. The output is a data frame where each
    row represents a single pixel value, with an index value identifying
    unique digit samples.
    """
    return (
        pd.melt(df.reset_index(), id_vars="index")
        .assign(
            **{
                "x": lambda x: x["variable"].str.extract(r"pixel_\d_(\d)").astype(int),
                "y": lambda x: x["variable"].str.extract(r"pixel_(\d)_\d").astype(int),
            }
        )
        .drop(columns=["variable"])
        .sort_values(["index", "x", "y"])
    )


def get_digits_id(row, index_col):
    """Get a unique identifier for the digit sample pixel."""
    return row[index_col].astype(str) + row["y"].astype(str) + row["x"].astype(str)


def fillna(df):
    """Fill missing values in the data frame."""
    # return df.fillna(-1)
    return df.map(lambda x: np.random.randint(0, 16) if pd.isnull(x) else x)


# Randomly corrupt a fraction of the training and test data.
X_train_corrupt = randomly_mask_values(X_train, frac=corrupt_frac, seed=0).pipe(fillna)
X_test_corrupt = randomly_mask_values(X_test, frac=corrupt_frac, seed=0).pipe(fillna)

# We set `max_samples_leaf=None` so that all leaf node samples are stored.
qrf = RandomForestQuantileRegressor(max_samples_leaf=None, random_state=0)
qrf.fit(X_train_corrupt, X_train)

proximities = qrf.proximity_counts(X_test_corrupt)

df_train = (
    digits_wide_to_long(X_train.reset_index(drop=True))
    .merge(y_train.reset_index(drop=True).reset_index(), on="index", how="left")
    .assign(**{"samp_id": lambda x: get_digits_id(x, "index")})
    .drop(columns=["index"])
)

df_test = digits_wide_to_long(X_test).merge(y_test.reset_index(), on="index", how="left")
df_test_corrupt = digits_wide_to_long(X_test_corrupt).rename(columns={"value": "value_corrupt"})

df_prox = pd.DataFrame(
    {
        "prox": [[(j, *p) for j, p in enumerate(proximities[i])] for i in range(n_test_samples)],
        "index": X_test.index,
    }
)

df = (
    df_test.merge(df_test_corrupt, on=["index", "x", "y"], how="left")
    .merge(df_prox, on="index", how="left")
    .rename(columns={"index": "samp_idx"})
    .explode("prox")
    .assign(
        **{
            "index": lambda x: pd.factorize(x["samp_idx"])[0],
            "prox_idx": lambda x: x["prox"].apply(lambda y: y[0]),
            "prox_val": lambda x: x["prox"].apply(lambda y: y[1]),
            "prox_count": lambda x: x["prox"].apply(lambda y: y[2]),
            "samp_id": lambda x: get_digits_id(x, "samp_idx"),
            "prox_id": lambda x: get_digits_id(x, "prox_val"),
        }
    )
    .drop(columns=["prox"])
)


def plot_digits_proximities(
    df,
    n_prox=25,
    n_prox_per_row=5,
    subplot_spacing=10,
    height=225,
    width=225,
):
    n_samples = df["samp_idx"].nunique()
    n_subplot_rows = n_prox // n_prox_per_row
    subplot_dim = (width - subplot_spacing * (2 + max(n_subplot_rows - 2, 0))) / n_subplot_rows

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

    color = alt.Color("value:Q", legend=None, scale=alt.Scale(scheme="greys"))
    opacity = alt.condition(alt.datum["value"] == 0, alt.value(0), alt.value(1))

    base = alt.Chart(df).add_params(idx_val).transform_filter(idx_val)

    chart1 = (
        base.mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=alt.Color("value_corrupt:Q", legend=None, scale=alt.Scale(scheme="greys")),
            # opacity=alt.condition(alt.datum["value_corrupt"] == -1, alt.value(0), alt.value(1)),
            opacity=alt.condition(alt.datum["value_corrupt"] == 0, alt.value(0), alt.value(1)),
            tooltip=[
                alt.Tooltip("target:Q", title="Digit"),
                alt.Tooltip("value_corrupt:Q", title="Pixel Value"),
                alt.Tooltip("value:Q", title="Pixel Value (original)"),
                alt.Tooltip("x:Q", title="Pixel X"),
                alt.Tooltip("y:Q", title="Pixel Y"),
            ],
        )
        .properties(height=height, width=width, title="Test Digit (corrupted)")
    )

    chart2_i = (
        base.mark_rect()
        .encode(
            x=alt.X("x:N", axis=None),
            y=alt.Y("y:N", axis=None),
            color=color,
            opacity=opacity,
            tooltip=[
                alt.Tooltip("prox_count", title="Training Proximity Count"),
                alt.Tooltip("target:Q", title="Digit"),
            ],
        )
        .transform_lookup(
            lookup="prox_id",
            from_=alt.LookupData(df_train, key="samp_id", fields=["x", "y", "target", "value"]),
        )
        .properties(height=subplot_dim, width=subplot_dim)
    )

    chart2_plots = [chart2_i.transform_filter(f"datum.prox_idx == {i}") for i in range(n_prox)]
    chart2_rows = [chart2_plots[i : i + n_prox_per_row] for i in range(0, n_prox, n_prox_per_row)]

    chart2 = alt.hconcat()
    for row in chart2_rows:
        rowplot = alt.vconcat()
        for item in row:
            rowplot |= item
        chart2 &= rowplot
    chart2 = chart2.properties(title=f"Proximity Digits (top {n_prox})")

    chart3 = chart1.encode(color=color, opacity=opacity).properties(title="Test Digit (original)")

    chart_spacer = alt.Chart(pd.DataFrame()).mark_rect().properties(width=subplot_dim)

    chart = (
        (chart1 | chart_spacer | chart2 | chart_spacer | chart3)
        .configure_concat(spacing=subplot_spacing)
        .configure_title(anchor="middle")
        .configure_view(strokeOpacity=0)
    )

    return chart


chart = plot_digits_proximities(df)
chart
