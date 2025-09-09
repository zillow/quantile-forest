"""
Using a Trained QRF Model via Hugging Face Hub
==============================================

This example demonstrates how to download a trained quantile regression forest
(QRF) model from Hugging Face Hub and use it to estimate quantiles. In this
scenario, a QRF has been trained with default parameters on the California
Housing dataset using k-fold cross-validation and uploaded to Hugging Face
Hub. The model is downloaded and used to perform inference across multiple
quantiles for each sample in the dataset. The predictions are aggregated by
county based on the latitude and longitude of each sample and visualized.
The trained model is available on Hugging Face Hub
`here <https://huggingface.co/quantile-forest/california-housing-example>`_.
"""

import os
import shutil
import tempfile

import altair as alt
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn import datasets
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from vega_datasets import data

try:
    from examples import hub_utils
except ImportError:
    from __init__ import hub_utils

token = "<Hugging Face Access Token>"
repo_id = "quantile-forest/california-housing-example"
load_existing = True

random_state = np.random.RandomState(0)
quantiles = np.linspace(0, 1, num=21, endpoint=True).round(2).tolist()


class CrossValidationPipeline(BaseEstimator, RegressorMixin):
    """Cross-validation pipeline for scikit-learn compatible models."""

    def __init__(self, base_model, n_splits=5, random_state=None):
        self.base_model = base_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.fold_models = {}
        self.fold_indices = {}
        self.is_fitted = False

    def fit(self, X, y):
        """Fit the model using k-fold cross-validation."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            model = clone(self.base_model)
            model.fit(X_train, y_train)
            self.fold_models[fold_idx] = model
            self.fold_indices[fold_idx] = test_idx
        self.is_fitted = True
        return self

    def predict(self, X, quantiles=None):
        """Predict using the appropriate k-fold model."""
        if quantiles is None:
            quantiles = 0.5
        if not isinstance(quantiles, list):
            quantiles = [quantiles]
        y_pred = np.empty((X.shape[0], len(quantiles)) if len(quantiles) > 1 else (X.shape[0]))
        for fold_idx, test_idx in self.fold_indices.items():
            fold_model = self.fold_models[fold_idx]
            y_pred[test_idx] = fold_model.predict(X.iloc[test_idx], quantiles=quantiles)
        return y_pred

    def save(self, filename):
        with open(filename, "wb") as f:
            joblib.dump(self.__getstate__(), f)

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            state = joblib.load(f)
        obj = cls(base_model=None)
        obj.__setstate__(state)
        return obj


def fit_and_upload_model(token, repo_id, local_dir="./local_repo", random_state=None):
    """Function used to fit the model and upload it to Hugging Face Hub."""
    from pathlib import Path

    from sklearn.metrics import (
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.pipeline import Pipeline
    from skops import card

    import quantile_forest
    from quantile_forest import RandomForestQuantileRegressor

    # Load the California Housing dataset.
    X, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)

    # Define the model pipeline.
    qrf = RandomForestQuantileRegressor(random_state=random_state)
    pipeline = Pipeline(
        [("cv_model", CrossValidationPipeline(qrf, n_splits=5, random_state=random_state))]
    )

    # Fit the model pipeline.
    pipeline.fit(X, y)

    # Save the pipeline (with all models) to a file.
    model_filename = "model.pkl"
    pipeline.named_steps["cv_model"].save(model_filename)

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
        data=X,
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
        "This is a RandomForestQuantileRegressor trained on the California Housing dataset."
    )
    limitations = "This model is not ready to be used in production."
    training_procedure = (
        "The model was trained using default parameters on a 5-fold cross-validation pipeline."
    )
    get_started_code = """<details>
<summary> Click to expand </summary>

```python
from examples.plot_qrf_huggingface_inference import CrossValidationPipeline
pipeline = CrossValidationPipeline.load(qrf_pkl_filename)
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
    y_pred = pipeline.predict(X)
    mape = mean_absolute_percentage_error(y, y_pred)
    mdae = median_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
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
    try:
        hub_utils.push(
            repo_id=repo_id,
            source=local_dir,
            token=token,  # personal token to be downloaded from Hugging Face
            commit_message="Model commit",
            create_remote=True,
        )
    except Exception as e:
        print(f"Error pushing model to Hugging Face Hub: {e}")

    os.remove(model_filename)
    shutil.rmtree(local_dir)


if not load_existing:
    fit_and_upload_model(token, repo_id, random_state=random_state)

# Download the repository locally and load the fitted model.
model_filename = "model.pkl"
local_dir = "./local_repo"
with tempfile.TemporaryDirectory() as local_dir:
    hub_utils.download(repo_id=repo_id, dst=local_dir)
    pipeline = CrossValidationPipeline.load(f"{local_dir}/{model_filename}")

# Fetch the California Housing dataset and estimate quantiles.
X, y = datasets.fetch_california_housing(as_frame=True, return_X_y=True)
y_pred = pipeline.predict(X, quantiles=quantiles) * 100_000  # predict in dollars

df = (
    pd.DataFrame(y_pred, columns=quantiles)
    .reset_index()
    .rename(columns={q: f"q_{q:.3g}" for q in quantiles})
    .merge(X[["Latitude", "Longitude", "Population"]].reset_index(), on="index", how="right")
)


def plot_quantiles_by_latlon(df, quantiles, color_scheme="lightgreyred"):
    """Plot quantile predictions on California Housing dataset by lat/lon."""
    # Slider for varying the displayed quantile estimates.
    slider = alt.binding_range(
        name="Predicted Quantile: ",
        min=0,
        max=1,
        step=0.5 if len(quantiles) == 1 else 1 / (len(quantiles) - 1),
    )

    quantile_val = alt.param(name="quantile", value=0.5, bind=slider)

    # Download the us-10m.json file temporarily.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        tmpfile.write(requests.get(data.us_10m.url).content)
        tmp_path = tmpfile.name

    # Load the US counties data and filter to California counties.
    ca_counties = (
        gpd.read_file(f"TopoJSON:{tmp_path}", layer="counties")
        .set_crs("EPSG:4326")
        .assign(**{"county_fips": lambda x: x["id"].astype(int)})
        .drop(columns=["id"])
        .query("(county_fips >= 6000) & (county_fips < 7000)")
    )

    x_min = df[[f"q_{q:.3g}" for q in quantiles]].min().min()
    x_max = df[[f"q_{q:.3g}" for q in quantiles]].max().max()

    df = (
        gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="4326"
        )
        .sjoin(ca_counties, how="right")
        .drop(columns=["index_left0"])
        .assign(
            **{f"w_q_{q:.3g}": lambda x, q=q: x[f"q_{q:.3g}"] * x["Population"] for q in quantiles}
        )
    )

    grouped = (
        df.groupby("county_fips")
        .agg({**{f"w_q_{q:.3g}": "sum" for q in quantiles}, **{"Population": "sum"}})
        .reset_index()
        .assign(
            **{f"q_{q:.3g}": lambda x, q=q: x[f"w_q_{q:.3g}"] / x["Population"] for q in quantiles}
        )
    )

    df = (
        df[["county_fips", "Latitude", "Longitude", "geometry"]]
        .drop_duplicates(subset=["county_fips"])
        .merge(
            grouped[["county_fips", "Population"] + [f"q_{q:.3g}" for q in quantiles]],
            on="county_fips",
            how="left",
        )
    )

    chart = (
        alt.Chart(df)
        .add_params(quantile_val)
        .transform_calculate(quantile_col="'q_' + quantile")
        .transform_calculate(value="datum[datum.quantile_col]")
        .mark_geoshape(stroke="black", strokeWidth=0.5)
        .encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(domain=[x_min, x_max], scheme=color_scheme),
                title="Prediction",
            ),
            tooltip=[
                alt.Tooltip("county_fips:N", title="County FIPS"),
                alt.Tooltip("Population:N", format=",.0f", title="Population"),
                alt.Tooltip("value:Q", format="$,.0f", title="Predicted Value"),
            ],
        )
        .project(type="mercator")
        .properties(
            title="Quantile Predictions on the California Housing Dataset by County",
            height=650,
            width=650,
        )
    )
    return chart


chart = plot_quantiles_by_latlon(df, quantiles)
chart
