"""
SHAP Explanations with Quantile Regression Forests
==================================================

This example demonstrates the use of SHAP (SHapley Additive exPlanations) to
explain the predictions of a quantile regression forest (QRF) model. SHAP
values offer insight into the contribution of each feature to the predicted
value. We generate a waterfall plot to visualize the explanations for a single
sample across multiple quantiles. The `Tree SHAP
<https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html>`_
method explains predictions that are sums over the trees of a model, which the
quantile predictions of a QRF generally are not; by default, a quantile is
calculated from the training samples pooled across the trees. Instead, we use
the `exact explainer
<https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html>`_,
which explains any model from its predictions. The SHAP values add up to the
predicted quantile, and this plot allows us to observe how the feature
contributions vary with the quantile, since a feature can change the spread of
the predicted distribution, not just its location. With fully grown trees (the
default), the low and high quantiles come from few training samples, which
makes their SHAP values noisy, so the leaves here keep at least 10 samples,
and there are 300 trees.
"""

import altair as alt
import numpy as np
import pandas as pd
import shap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from quantile_forest import RandomForestQuantileRegressor

random_state = np.random.RandomState(0)
n_samples = 1000
test_idx = 0
quantiles = np.linspace(0.1, 0.9, num=9).round(1).tolist()  # in steps of 0.1

# Load the California Housing dataset.
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
perm = random_state.permutation(min(len(X), n_samples))
X = X.iloc[perm]
y = y.iloc[perm]
y *= 100_000  # convert to dollars

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

# Keep at least 10 training samples in each leaf, and fit more trees than the
# default, for quantiles (and SHAP values) that vary less from fit to fit.
qrf = RandomForestQuantileRegressor(
    n_estimators=300, min_samples_leaf=10, random_state=random_state
)
qrf.fit(X_train, y_train)


def predict_quantiles(X):
    """Predict the quantiles (with any other parameters of `predict`, too)."""
    return qrf.predict(X, quantiles=quantiles)


# Get the SHAP values at each quantile for the specified test sample, with the
# features outside of each subset taken from the training data.
masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
explainer = shap.explainers.Exact(predict_quantiles, masker)
X_explain = X_test.iloc[[test_idx]]
shap_values = explainer(X_explain, silent=True)[0]  # the SHAP values by feature and quantile

# The SHAP values add up to the QRF predictions.
y_pred = predict_quantiles(X_explain)[0]
np.testing.assert_allclose(shap_values.base_values + shap_values.values.sum(axis=0), y_pred)

df = pd.concat(
    [
        pd.DataFrame(
            {
                "feature": [f"{round(v, 3):g} = {c}" for c, v in X_test.iloc[test_idx].items()],
                "feature_name": X.columns,
                "feature_value": X_test.iloc[test_idx].to_numpy(),
                "shap_value": shap_values.values[:, i],
                "abs_shap_value": abs(shap_values.values[:, i]),
                "base_value": shap_values.base_values[i],
                "model_output": shap_values.base_values[i] + sum(shap_values.values[:, i]),
                "quantile": q,
            }
        )
        for i, q in enumerate(quantiles)
    ],
    ignore_index=True,
)


def plot_shap_waterfall_with_quantiles(df, height=300):
    """Plot SHAP waterfall plot by quantile predictions."""
    df = df.copy()

    # Slider for varying the applied quantile estimates.
    slider = alt.binding_range(
        name="Predicted Quantile: ", min=quantiles[0], max=quantiles[-1], step=0.1
    )
    quantile_val = alt.param(name="quantile", value=0.5, bind=slider)

    df_grouped = (
        df.groupby("quantile")[df.columns.tolist()]
        .apply(lambda g: g.sort_values("abs_shap_value", ascending=True))
        .reset_index(drop=True)
        .assign(
            **{
                "start": (
                    lambda df: df.groupby("quantile", group_keys=False).apply(
                        lambda g: g["shap_value"].shift(1, fill_value=0).cumsum()
                        + g["base_value"],
                        include_groups=False,
                    )
                ),
                "end": (
                    lambda df: df.groupby("quantile", group_keys=False).apply(
                        lambda g: g["shap_value"].cumsum() + g["base_value"], include_groups=False
                    )
                ),
                "value_label": (
                    lambda df: df["shap_value"].apply(
                        lambda x: ("+" if x >= 0 else "-") + "{0:,.2f}".format(abs(x))
                    )
                ),
                "feature2": (
                    lambda df: df.groupby("quantile", group_keys=False).apply(
                        lambda g: g["feature"].shift(-1), include_groups=False
                    )
                ),
            }
        )
        .groupby("quantile")[df.columns.tolist() + ["start", "end", "value_label", "feature2"]]
        .apply(lambda g: g.sort_values("abs_shap_value", ascending=False))
        .reset_index(drop=True)
    )

    x_min = df_grouped[["base_value", "model_output", "start", "end"]].min().min()
    x_max = df_grouped[["base_value", "model_output", "start", "end"]].max().max()
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
        .transform_filter("round(datum.quantile * 100) == round(quantile * 100)")
        .transform_calculate(
            end_shifted=f"datum.shap_value > 0 ? datum.end - {x_shift} : datum.end + {x_shift}"
        )
        .transform_calculate(
            end_shifted=f"abs(datum.shap_value) < {x_shift} ? datum.end : datum.end_shifted"
        )
    )

    bar = base.mark_bar().encode(
        x=alt.X(
            "start:Q",
            axis=alt.Axis(format=",.2f", grid=False),
            scale=alt.Scale(domain=[x_min - 10 * x_shift, x_max + 10 * x_shift], zero=False),
            title=None,
        ),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.condition(
            alt.datum["shap_value"] > 0, alt.value("#ff0251"), alt.value("#018bfb")
        ),
        tooltip=[
            alt.Tooltip("feature_name:N", title="Feature Name"),
            alt.Tooltip("feature_value:Q", title="Feature Value"),
            alt.Tooltip("shap_value:Q", format=",.2f", title="SHAP Value"),
            alt.Tooltip("start:Q", format=",.2f", title="SHAP Start"),
            alt.Tooltip("end:Q", format=",.2f", title="SHAP End"),
        ],
    )

    point = (
        bar.transform_filter(f"abs(datum.shap_value) > {x_shift}")
        .mark_point(filled=True, opacity=1, size=125)
        .encode(
            x=alt.X("end_shifted:Q", title=None),
            shape=alt.condition(
                alt.datum["shap_value"] > 0, alt.value(triangle_right), alt.value(triangle_left)
            ),
        )
    )

    text_bar_left = bar.mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="gray",
    ).encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(0), alt.value(1)),
    )
    text_bar_right = bar.mark_text(align="right", baseline="middle", dx=-5, color="gray").encode(
        text="value_label",
        opacity=alt.condition(alt.datum["shap_value"] > 0, alt.value(1), alt.value(0)),
    )
    text_label_start = (
        alt.Chart(df_text_labels)
        .transform_filter("round(datum.quantile * 100) == round(quantile * 100)")
        .transform_filter("datum.type == 'start'")
        .mark_text(align="left", color="gray", dx=-16, dy=y_text_offset + 30)
        .encode(text=alt.Text("label"), x=alt.X("x:Q"))
    )
    text_label_end = (
        alt.Chart(df_text_labels)
        .transform_filter("round(datum.quantile * 100) == round(quantile * 100)")
        .transform_filter("datum.type == 'end'")
        .mark_text(align="left", color="gray", dx=-8, dy=-y_text_offset - 15)
        .encode(text=alt.Text("label"), x=alt.X("x:Q"))
    )
    text = text_bar_left + text_bar_right + text_label_start + text_label_end

    feature_bar_rule = (
        base.transform_filter("isValid(datum.feature2)")
        .mark_rule(
            color="gray",
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
        .transform_filter("round(datum.quantile * 100) == round(quantile * 100)")
        .transform_filter("datum.type == 'start'")
        .mark_rule(color="gray", opacity=1, y=height, y2=height + 6)
        .encode(x=alt.X("x:Q"))
    )
    tick_end_rule = (
        alt.Chart(df_text_labels)
        .transform_filter("round(datum.quantile * 100) == round(quantile * 100)")
        .transform_filter("datum.type == 'end'")
        .mark_rule(color="gray", opacity=1, y=0, y2=-6)
        .encode(x=alt.X("x:Q"))
    )
    rule = feature_bar_rule + end_bar_rule + tick_start_rule + tick_end_rule

    bar = bar.encode(x2=alt.X2("end_shifted:Q"))

    chart = (
        (bar + point + text + rule)
        .add_params(quantile_val)
        .configure_view(strokeOpacity=0)
        .properties(
            title="Waterfall Plot of SHAP Values for QRF Predictions", height=height, width=600
        )
    )

    return chart


chart = plot_shap_waterfall_with_quantiles(df)
chart
