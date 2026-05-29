import os
import json
import argparse
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

import category_encoders as ce
import xgboost as xg


# =====================================================
# Configuration
# =====================================================
RANDOM_STATE = 43
TARGET_COL = "Profit"

DF_CAT = [
    "Category",
    "City",
    "Country",
    "Order ID",
    "Order Priority",
    "Product ID",
    "Product Name",
    "Sub-Category",
]

DF_NUM = [
    "Discount",
    "Sales",
    "Shipping Cost",
    "weeknum",
]

FEATURE_COLUMNS = DF_NUM + DF_CAT
NEGATIVE_CHECK_COLUMNS = DF_NUM + [TARGET_COL]


# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================================
# Data Loading
# =====================================================
def load_data(input_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Parquet file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Only .csv and .parquet files are supported")

    logger.info("Loaded dataset with shaps: %s", df.shape)
    
    return df


# =====================================================
# Validation
# =====================================================
def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate required columns and empty dataset.
    """
    required_cols = set(FEATURE_COLUMNS + [TARGET_COL])
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input dataset is empty")

    logger.info("Dataset validation passed.")


# =====================================================
# Preprocessing / Cleaning
# =====================================================
def preprocess_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean data before training:
    1. Keep only required columns
    2. Convert numeric columns safely
    3. Remove rows with missing required values
    4. Remove rows having negative values in numeric columns or target
    5. Clean categorical text values
    """
    df = df[FEATURE_COLUMNS + [TARGET_COL]].copy()
    initial_rows = len(df)

    # Convert numeric columns safely
    numeric_all = DF_NUM + [TARGET_COL]
    for col in numeric_all:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical columns
    for col in DF_CAT:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df[col] = df[col].replace("", "Unknown")

    # Remove rows with missing numeric / target
    before_missing_drop = len(df)
    df = df.dropna(subset=numeric_all)
    removed_missing = before_missing_drop - len(df)

    # Remove rows with negative numeric / target values
    negative_mask = (df[NEGATIVE_CHECK_COLUMNS] < 0).any(axis=1)
    removed_negative = int(negative_mask.sum())
    df = df.loc[~negative_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "All rows were removed during preprocessing. "
            "Please check whether your dataset contains only negative or invalid values."
        )

    summary = {
        "initial_rows": int(initial_rows),
        "removed_missing_numeric_or_target": int(removed_missing),
        "removed_negative_rows": int(removed_negative),
        "final_rows": int(len(df)),
    }

    logger.info("Preprocessing summary: %s", summary)
    return df, summary


# =====================================================
# Pipeline Builder
# =====================================================
def build_pipeline() -> Pipeline:
    """
    Build preprocessing + feature selection + model pipeline.
    """
    transform_num_data = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
    ])

    transform_cat_data = Pipeline(steps=[
        ("encode", ce.CatBoostEncoder(handle_unknown="value", handle_missing="value")),
        ("power", PowerTransformer(method="yeo-johnson")),
    ])

    process = ColumnTransformer(
        transformers=[
            ("number", transform_num_data, DF_NUM),
            ("category", transform_cat_data, DF_CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    df_select_columns = RFECV(
        estimator=xg.XGBRFRegressor(
            random_state=RANDOM_STATE,
            n_estimators=150,
            n_jobs=-1,
        ),
        step=0.1,
        cv=5,
        scoring="r2",
        min_features_to_select=8,
        n_jobs=-1,
    )

    model = Pipeline(steps=[
        ("process", process),
        ("select_col", df_select_columns),
        ("ref", RandomForestRegressor(
            ccp_alpha=0.01,
            criterion="squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True,
            n_estimators=150,
        )),
    ])

    return model


# =====================================================
# Evaluation
# =====================================================
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# =====================================================
# Metadata
# =====================================================
def build_metadata(df: pd.DataFrame, metrics: dict, preprocess_summary: dict) -> dict:
    """
    Build metadata for Streamlit application.
    """
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "target": TARGET_COL,
        "numeric_columns": DF_NUM,
        "categorical_columns": DF_CAT,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "preprocess_summary": preprocess_summary,
        "negative_rows_removed_during_training": True,
        "ui_defaults": {
            "numeric_defaults": {
                col: float(pd.to_numeric(df[col], errors="coerce").median())
                for col in DF_NUM
            },
            "categorical_options": {
                col: sorted(df[col].dropna().astype(str).unique().tolist())
                for col in DF_CAT
            },
        },
    }
    return metadata


# =====================================================
# Save Artifact
# =====================================================
def save_artifact(model: Pipeline, metadata: dict, output_path: str) -> None:
    """
    Save model artifact.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    artifact = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(artifact, output_path,compress=("lzma", 9))
    logger.info("Saved model artifact to: %s", output_path)


# =====================================================
# Main
# =====================================================
def main(input_path: str, output_path: str, test_size: float) -> None:
    logger.info("Starting model training...")

    df = load_data(input_path)
    validate_dataframe(df)

    # Preprocess and remove negative rows
    df_clean, preprocess_summary = preprocess_training_data(df)

    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    model = build_pipeline()

    logger.info("Fitting model...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metadata = build_metadata(df_clean, metrics, preprocess_summary)

    save_artifact(model, metadata, output_path)

    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete.")
    logger.info("Model artifact saved at: %s", output_path)
    logger.info("Metrics JSON saved at: %s", metrics_path)


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save profit prediction model after removing negative values"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV or Parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/profit_model.joblib",
        help="Path to output model artifact"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.30,
        help="Test size ratio"
    )

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_path=args.output,
        test_size=args.test_size
    )