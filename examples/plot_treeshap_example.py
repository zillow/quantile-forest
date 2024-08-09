"""
Tree SHAP with Quantile Regression Forests
==========================================

This example demonstrates the use of SHAP (SHapley Additive exPlanations) to
explain the predictions of a quantile regression forest (QRF) model. We
generate a waterfall plot using the `Tree SHAP
<https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#shap.TreeExplainer>`_
method to visualize the explanations for a single instance across multiple
quantiles. In a QRF, quantile estimation is applied during inference, meaning
the selected quantile affects the expected value of the model output but does
not alter the feature contributions. This plot allows us to observe how the
SHAP explanations vary with different quantile choices.
"""

import altair as alt
import numpy as np
import pandas as pd
import shap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_seed = 0
n_samples = 500
test_idx = 0
quantiles = np.linspace(0, 1, num=11, endpoint=True).round(1).tolist()


def get_shap_values(qrf, X, quantile=0.5, **kwargs):
    # Define a custom tree model.
    model = {
        "objective": qrf.criterion,
        "tree_output": "raw_value",
        "trees": [e.tree_ for e in qrf.estimators_],
    }

    # Use Tree SHAP to generate explanations.
    explainer = shap.TreeExplainer(model, X)

    qrf_pred = qrf.predict(X, quantiles=quantile, **kwargs)
    rf_pred = qrf.predict(X, quantiles="mean", aggregate_leaves_first=False)

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
X = X.iloc[:n_samples]
y = y[:n_samples]
y *= 100_000  # convert to dollars
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)

qrf = RandomForestQuantileRegressor(random_state=random_seed)
qrf.fit(X_train, y_train)

df = pd.concat(
    [
        pd.DataFrame(
            {
                "feature": [f"{X.iloc[test_idx, i]} = {X.columns[i]}" for i in range(X.shape[1])],
                "feature_name": X.columns,
                "feature_value": [f"{X.iloc[test_idx, i]}" for i in range(X.shape[1])],
                "shap_value": shap_i.values,
                "abs_shap_value": abs(shap_i.values),
                "base_value": shap_i.base_values,
                "model_output": shap_i.base_values + sum(shap_i.values),
                "quantile": q,
            }
        )
        for q in quantiles
        for shap_i in [get_shap_value_by_index(get_shap_values(qrf, X_test, quantile=q), test_idx)]
    ],
    ignore_index=True,
)


def plot_shap_waterfall_with_quantiles(df, height=300):
    df = df.copy()

    # Slider for varying the applied quantile estimates.
    slider = alt.binding_range(
        min=0,
        max=1,
        step=0.5 if len(quantiles) == 1 else 1 / (len(quantiles) - 1),
        name="Predicted Quantile: ",
    )
    quantile_selection = alt.param(value=0.5, bind=slider, name="quantile")

    df_grouped = (
        df.groupby("quantile")[df.columns.tolist()]
        .apply(lambda g: g.sort_values("abs_shap_value", ascending=True))
        .reset_index(drop=True)
        .assign(
            **{
                "start": lambda df: df.groupby("quantile", group_keys=False).apply(
                    lambda g: g["shap_value"].shift(1, fill_value=0).cumsum() + g["base_value"],
                    include_groups=False,
                ),
                "end": lambda df: df.groupby("quantile", group_keys=False).apply(
                    lambda g: g["shap_value"].cumsum() + g["base_value"], include_groups=False
                ),
                "value_label": lambda df: df["shap_value"].apply(
                    lambda x: ("+" if x >= 0 else "-") + "{0:,.2f}".format(abs(x))
                ),
                "feature2": lambda df: df.groupby("quantile", group_keys=False).apply(
                    lambda g: g["feature"].shift(-1), include_groups=False
                ),
            }
        )
        .groupby("quantile")[df.columns.tolist() + ["start", "end", "value_label", "feature2"]]
        .apply(lambda g: g.sort_values("abs_shap_value", ascending=False))
        .reset_index(drop=True)
    )

    x_min = min(df["base_value"].min(), df["model_output"].min())
    x_max = max(df["base_value"].max(), df["model_output"].max())
    x_shift = (x_max - x_min) / 100

    y_rule_offset = round(height / df_grouped["feature"].nunique() / 2) - 2
    y_text_offset = round(height / 2)

    triangle_size = round(height / 100)
    triangle_left = f"M 0,{triangle_size} L -1,0 L 0,-{triangle_size} Z"
    triangle_right = f"M 0,-{triangle_size} L 1,0 L 0,{triangle_size} Z"

    df_text_labels = (
        df_grouped.groupby("quantile")[df.columns]
        .apply(
            lambda g: pd.DataFrame(
                {
                    "label": [
                        f"f(x) = {g['model_output'].iloc[0]:,.2f}",
                        f"E[f(X)] = {g['base_value'].iloc[0]:,.2f}",
                    ],
                    "type": ["end", "start"],
                    "x": [g["model_output"].iloc[0], g["base_value"].iloc[0]],
                    "quantile": [g["quantile"].iloc[0], g["quantile"].iloc[0]],
                }
            )
        )
        .reset_index(drop=True)
    )

    base = (
        alt.Chart(df_grouped)
        .transform_filter("datum.quantile == quantile")
        .transform_calculate(
            end_shifted=f"datum.shap_value > 0 ? datum.end - {x_shift} : datum.end + {x_shift}"
        )
        .transform_calculate(
            end_shifted=f"abs(datum.shap_value) < {x_shift} ? datum.end : datum.end_shifted"
        )
    )

    bars = base.mark_bar().encode(
        x=alt.X(
            "start:Q",
            axis=alt.Axis(format=",.2f", grid=False),
            scale=alt.Scale(domain=[x_min, x_max], zero=False),
            title=None,
        ),
        x2=alt.X2("end_shifted:Q"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.condition(
            alt.datum["shap_value"] > 0, alt.value("#ff0251"), alt.value("#018bfb")
        ),
        tooltip=[
            alt.Tooltip("feature_name:N", title="Feature"),
            alt.Tooltip("shap_value:Q", format=",.2f", title="SHAP Value"),
            alt.Tooltip("start:Q", format=",.2f", title="SHAP Start"),
            alt.Tooltip("end:Q", format=",.2f", title="SHAP End"),
        ],
    )

    points = (
        bars.transform_filter(f"abs(datum.shap_value) > {x_shift}")
        .mark_point(filled=True, opacity=1, size=125)
        .encode(
            x=alt.X("end_shifted:Q", title=None),
            shape=alt.condition(
                alt.datum["shap_value"] > 0, alt.value(triangle_right), alt.value(triangle_left)
            ),
        )
    )

    text_bar_left = bars.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="black",
    ).encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(0), alt.value(1)),
    )
    text_bar_right = bars.mark_text(align="right", baseline="middle", dx=-5, color="black").encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(1), alt.value(0)),
    )
    text_label_start = (
        alt.Chart(df_text_labels)
        .transform_filter("datum.quantile == quantile")
        .transform_filter("datum.type == 'start'")
        .mark_text(align="left", color="black", dx=-16, dy=y_text_offset + 30)
        .encode(text=alt.Text("label"), x=alt.X("x:Q"))
    )
    text_label_end = (
        alt.Chart(df_text_labels)
        .transform_filter("datum.quantile == quantile")
        .transform_filter("datum.type == 'end'")
        .mark_text(align="left", color="black", dx=-8, dy=-y_text_offset - 15)
        .encode(text=alt.Text("label"), x=alt.X("x:Q"))
    )
    text = text_bar_left + text_bar_right + text_label_start + text_label_end

    feature_bar_rule = (
        base.transform_filter("isValid(datum.feature2)")
        .mark_rule(
            color="black",
            yOffset=y_rule_offset,
            y2Offset=-y_rule_offset,
            opacity=0.8,
            strokeDash=[1, 1],
        )
        .encode(
            x=alt.X("end:Q"),
            y=alt.Y("feature", sort=None),
            y2=alt.Y2("feature2"),
        )
    )
    end_bar_rule = base.mark_rule(color="gray", opacity=0.8, strokeDash=[1, 1]).encode(
        x=alt.X("model_output:Q")
    )
    tick_start_rule = (
        alt.Chart(df_text_labels)
        .transform_filter("datum.quantile == quantile")
        .transform_filter("datum.type == 'start'")
        .mark_rule(color="black", opacity=1, y=height, y2=height + 6)
        .encode(x=alt.X("x:Q"))
    )
    tick_end_rule = (
        alt.Chart(df_text_labels)
        .transform_filter("datum.quantile == quantile")
        .transform_filter("datum.type == 'end'")
        .mark_rule(color="black", opacity=1, y=0, y2=-6)
        .encode(x=alt.X("x:Q"))
    )
    rules = feature_bar_rule + end_bar_rule + tick_start_rule + tick_end_rule

    chart = (
        (bars + points + text + rules)
        .add_params(quantile_selection)
        .configure_view(strokeOpacity=0)
        .properties(
            width=600, height=height, title="Waterfall Plot of SHAP Values for QRF Predictions"
        )
    )

    return chart


chart = plot_shap_waterfall_with_quantiles(df)
chart
