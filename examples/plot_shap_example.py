"""
Tree SHAP with Quantile Regression Forests
==========================================

An example of using SHAP to explain predictions generated by a quantile
regression forest model. Here, we generate a waterfall plot of the Tree SHAP
explanations for a single instance across several quantiles. This plot helps
us understand how the explanations change with different quantile selections.
"""

import altair as alt
import numpy as np
import pandas as pd
import shap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor


def get_shap_values(qrf, X, quantile=0.5, **kwargs):
    # Define a custom tree model.
    model = {
        "objective": qrf.criterion,
        "tree_output": "raw_value",
        "trees": [e.tree_ for e in qrf.estimators_],
    }

    # Use Tree SHAP to generate explanations.
    explainer = shap.TreeExplainer(model, X)

    qrf_pred = qrf.predict(X.to_numpy(), quantiles=quantile, **kwargs)
    rf_pred = qrf.predict(X.to_numpy(), quantiles="mean", aggregate_leaves_first=False)

    scaling = 1.0 / len(qrf.estimators_)  # scale factor based on the number of estimators
    base_offset = qrf_pred - rf_pred  # difference between the QRF and RF (baseline) predictions

    # Adjust the tree model values.
    explainer.model.values *= scaling  # multiply based on the scaling

    # Adjust the explainer expected value.
    explainer.expected_value *= scaling  # multiply based on the scaling
    explainer.expected_value = np.tile(explainer.expected_value, len(X))  # tile to length of X
    explainer.expected_value += np.array(base_offset)  # adjust based on the quantile

    shap_values = explainer(X, check_additivity=False)
    shap_values.base_values = np.diag(shap_values.base_values)

    return shap_values


def get_shap_value_by_index(shap_values, index):
    shap_values_i = shap_values[index]
    shap_values_i.base_values = shap_values.base_values[index]
    return shap_values_i


# Load the California Housing Prices dataset.
X, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
X = X.iloc[:500]
y = y[:500]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=0)

qrf = RandomForestQuantileRegressor(random_state=0)
qrf.fit(X_train, y_train)

test_idx = 0
quantiles = list((np.arange(11) * 10) / 100)

dfs = []
for quantile in quantiles:
    # Get the SHAP values for the test data.
    shap_values = get_shap_values(qrf, X_test, quantile=quantile)

    # Get the SHAP values for a particular test instance (by index).
    shap_values_by_idx = get_shap_value_by_index(shap_values, test_idx)

    dfs.append(
        pd.DataFrame(
            {
                "feature": [f"{X.iloc[test_idx, i]} = {X.columns[i]}" for i in range(X.shape[1])],
                "feature_name": X.columns,
                "shap_value": shap_values_by_idx.values,
                "abs_shap_value": abs(shap_values_by_idx.values),
                "base_value": shap_values_by_idx.base_values,
                "model_output": shap_values_by_idx.base_values + sum(shap_values_by_idx.values),
                "quantile": quantile,
            }
        )
    )
df = pd.concat(dfs)


def plot_shap_waterfall_with_quantiles(df):
    df = df.copy()

    # Slider for varying the applied quantile estimates.
    slider = alt.binding_range(
        min=0,
        max=1,
        step=0.5 if len(quantiles) == 1 else 1 / (len(quantiles) - 1),
        name="Quantile:",
    )

    q_val = alt.selection_point(
        value=0.5,
        bind=slider,
        fields=["quantile"],
    )

    df_grouped = (
        df.groupby("quantile")
        .apply(lambda x: x.sort_values("abs_shap_value", ascending=True))
        .reset_index(drop=True)
    )
    df_grouped["start"] = (
        df_grouped.groupby("quantile")
        .apply(lambda g: g["shap_value"].cumsum() + g["base_value"])
        .reset_index(drop=True)
    )
    df_grouped["end"] = (
        df_grouped.groupby("quantile")
        .apply(lambda g: g["shap_value"].shift(1, fill_value=0).cumsum() + g["base_value"])
        .reset_index(drop=True)
    )
    df_grouped["value_label"] = df_grouped["shap_value"].round(2).astype(str)
    df_grouped = (
        df_grouped.groupby("quantile")
        .apply(lambda x: x.sort_values("abs_shap_value", ascending=False))
        .reset_index(drop=True)
    )

    x_min = min(df["base_value"].min(), df["model_output"].min())
    x_max = max(df["base_value"].max(), df["model_output"].max())

    df_text_labels = (
        df_grouped.groupby("quantile")
        .apply(
            lambda g: pd.DataFrame(
                {
                    "label": [
                        f"f(X) = {round(g['model_output'].iloc[0], 3)}",
                        f"E[f(X)] = {round(g['base_value'].iloc[0], 3)}",
                    ],
                    "x": [g["model_output"].iloc[0], g["base_value"].iloc[0]],
                    "quantile": [g["quantile"].iloc[0], g["quantile"].iloc[0]],
                }
            )
        )
        .reset_index(drop=True)
    )

    base = alt.Chart(df_grouped).transform_filter(q_val)

    bars = base.mark_bar().encode(
        y=alt.Y("feature:N", sort=None, title="Feature"),
        x=alt.X(
            "end:Q",
            axis=alt.Axis(grid=False),
            scale=alt.Scale(domain=[x_min, x_max], zero=False),
            title="Value",
        ),
        x2=alt.X2("start:Q"),
        color=alt.condition(
            alt.datum["shap_value"] > 0, alt.value("#ff0251"), alt.value("#006aff")
        ),
        tooltip=[
            alt.Tooltip("feature_name:N", title="Feature"),
            alt.Tooltip("shap_value:Q", format=".3f", title="SHAP Value"),
            alt.Tooltip("start:Q", format=".3f", title="SHAP Start"),
            alt.Tooltip("end:Q", format=".3f", title="SHAP End"),
        ],
    )

    text_left = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="black",
    ).encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(0), alt.value(1)),
    )
    text_right = bars.mark_text(align="right", baseline="middle", dx=-5, color="black").encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(1), alt.value(0)),
    )
    text_labels = (
        alt.Chart(df_text_labels)
        .transform_filter(q_val)
        .mark_text(color="black", dx=0, dy=-210)
        .encode(text=alt.Text("label"), x=alt.X("x:Q"))
    )
    text = text_left + text_right + text_labels

    start_rule = base.mark_rule(color="black", opacity=0.8, strokeDash=[1, 1]).encode(
        x=alt.X("base_value:Q")
    )
    end_rule = base.mark_rule(color="gray", opacity=0.8, strokeDash=[1, 1]).encode(
        x=alt.X("model_output:Q")
    )

    chart = (
        (bars + text + start_rule + end_rule)
        .add_params(q_val)
        .properties(
            width=600, height=400, title="Waterfall Plot of SHAP Values for QRF Predictions"
        )
    )

    return chart


chart = plot_shap_waterfall_with_quantiles(df)
chart
