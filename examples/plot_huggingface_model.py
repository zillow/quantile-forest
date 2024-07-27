"""
Using a Trained QRF Model via Hugging Face Hub
==============================================

An example of downloading a trained quantile regression forest (QRF) model
from Hugging Face Hub and using it to estimate new quantiles. Here, a QRF has
been trained with default parameters on a train-test split of the California
housing dataset and uploaded to Hugging Face Hub. The model is downloaded and
inference is performed over several quantiles for each instance in the dataset.
The estimates are visualized by the latitude and longitude of each instance.
"""

import os
import pickle
import shutil

import altair as alt
import numpy as np
import pandas as pd
from sklearn import datasets
from skops import hub_utils

import quantile_forest
from quantile_forest import RandomForestQuantileRegressor

alt.data_transformers.disable_max_rows()

token = "<Hugging Face Access Token>"
repo_id = "quantile-forest/california-housing-example"
load_existing = True


def fit_and_upload_model(token, repo_id, local_dir="./local_repo"):
    """Function used to fit the model and upload it to Hugging Face Hub."""
    from pathlib import Path

    from sklearn.metrics import (
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from skops import card

    X, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Fit the model.
    qrf = RandomForestQuantileRegressor(random_state=0).fit(X_train, y_train)

    # Save the model to a file.
    model_filename = "model.pkl"
    with open(model_filename, mode="bw") as f:
        pickle.dump(qrf, file=f)

    # Prepare model repository.
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.mkdir(local_dir)

    # Initialize the repository.
    hub_utils.init(
        model=model_filename,
        requirements=[f"quantile-forest={quantile_forest.__version__}"],
        dst=local_dir,
        task="tabular-regression",
        data=X_train,
    )

    # Create a model card.
    model_card = card.Card(qrf, metadata=card.metadata_from_config(Path(local_dir)))
    model_card.metadata.library_name = "quantile-forest"
    model_card.metadata.license = "apache-2.0"
    model_card.metadata.tags = [
        "quantile-forest",
        "sklearn",
        "skops",
        "tabular-regression",
        "quantile-regression",
        "uncertainty-estimation",
        "prediction-intervals",
    ]
    model_description = (
        "This is a RandomForestQuantileRegressor trained on the California housing dataset."
    )
    limitations = "This model is not ready to be used in production."
    training_procedure = (
        "The model was trained using default parameters on a standard train-test split."
    )
    get_started_code = """<details>
<summary> Click to expand </summary>

```python
import pickle 
with open(qrf_pkl_filename, 'rb') as file: 
    qrf = pickle.load(file)
```

</details>"""
    model_card.add(
        **{
            "Model description": model_description,
            "Model description/Intended uses & limitations": limitations,
            "Model description/Training Procedure": training_procedure,
            "How to Get Started with the Model": get_started_code,
        }
    )

    # Add performance metrics to the model card.
    y_pred = qrf.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_card.add_metrics(
        **{
            "Mean Absolute Percentage Error": mape,
            "Median Absolute Error": mdae,
            "Mean Squared Error": mse,
            "R-Squared": r2,
        },
    )

    # Save the model card locally.
    model_card.save(Path(local_dir) / "README.md")

    # Create the repository on the Hugging Face Hub if it does not exist.
    # Push the model to the repository.
    hub_utils.push(
        repo_id=repo_id,
        source=local_dir,
        token=token,  # personal token to be downloaded from Hugging Face
        commit_message="Model commit",
        create_remote=True,
    )

    os.remove(model_filename)
    shutil.rmtree(local_dir)


def plot_quantiles_by_latlon(df, quantiles):
    # Slider for varying the displayed quantile estimates.
    slider = alt.binding_range(
        min=0,
        max=1,
        step=1 / (len(quantiles) - 1),
        name="Quantile:",
    )

    q_val = alt.selection_point(
        value=0.5,
        bind=slider,
        fields=["quantile"],
    )

    chart = (
        alt.Chart(df)
        .add_params(q_val)
        .transform_filter(f"datum['quantile'] == {q_val['quantile']}")
        .mark_circle()
        .encode(
            x=alt.X(
                "Longitude:Q",
                axis=alt.Axis(tickMinStep=1, format=".1f"),
                scale=alt.Scale(zero=False),
                title="Longitude",
            ),
            y=alt.Y(
                "Latitude:Q",
                axis=alt.Axis(tickMinStep=1, format=".1f"),
                scale=alt.Scale(zero=False),
                title="Latitude",
            ),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), title="Prediction"),
            size=alt.Size("Population:Q"),
            tooltip=[
                alt.Tooltip("index:N", title="Row ID"),
                alt.Tooltip("Latitude:Q", format=".2f", title="Latitude"),
                alt.Tooltip("Longitude:Q", format=".2f", title="Longitude"),
                alt.Tooltip("value:Q", format=",.0f", title="Predicted Value"),
            ],
        )
        .properties(
            height=650,
            width=650,
            title="Quantile Estimates on the California Housing Dataset",
        )
    )
    return chart


if not load_existing:
    fit_and_upload_model(token, repo_id)

# Download the repository locally.
local_dir = "./local_repo"
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)
hub_utils.download(repo_id=repo_id, dst=local_dir)

# Load the fitted model.
model_filename = "model.pkl"
with open(f"{local_dir}/{model_filename}", "rb") as file:
    qrf = pickle.load(file)
shutil.rmtree(local_dir)

# Estimate quantiles.
n_quantiles = 11
quantiles = list((np.arange(n_quantiles) * 10) / 100)
X, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
y_pred = qrf.predict(X, quantiles=quantiles) * 100_000  # predict in dollars

df = (
    pd.DataFrame(y_pred, columns=quantiles)
    .reset_index()
    .melt(id_vars=["index"], var_name="quantile", value_name="value")
    .merge(X[["Latitude", "Longitude", "Population"]].reset_index(), on="index", how="right")
)

chart = plot_quantiles_by_latlon(df, quantiles)
chart
