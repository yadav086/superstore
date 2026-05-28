import os
import io
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="Profit Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# Logging
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================================
# Constants
# =====================================================
MODEL_PATH = "artifacts/profit_model.joblib"
CUSTOM_OPTION_LABEL = "Other / Enter manually"


# =====================================================
# Load Artifact
# =====================================================
@st.cache_resource
def load_artifact(model_path: str):
    """
    Load saved model artifact.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Please run train.py first."
        )

    artifact = joblib.load(model_path)

    if "model" not in artifact or "metadata" not in artifact:
        raise ValueError("Artifact file is invalid. Expected keys: 'model' and 'metadata'.")

    return artifact


# =====================================================
# Validation Helpers
# =====================================================
def validate_input_df(df: pd.DataFrame, feature_columns: list, numeric_columns: list) -> pd.DataFrame:
    """
    Validate input dataframe for prediction.
    Reject negative numeric values because training data was cleaned to remove them.
    """
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[feature_columns].copy()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]

    # Convert numeric columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing numeric check
    if df[numeric_columns].isnull().any().any():
        bad_cols = df[numeric_columns].columns[df[numeric_columns].isnull().any()].tolist()
        raise ValueError(
            f"Numeric columns contain missing or invalid values: {bad_cols}. "
            "Please provide valid numeric input."
        )

    # Negative numeric check
    negative_mask = (df[numeric_columns] < 0)
    if negative_mask.any().any():
        negative_cols = negative_mask.columns[negative_mask.any()].tolist()
        raise ValueError(
            f"Negative values are not allowed in numeric columns: {negative_cols}. "
            "Please upload or enter only non-negative values."
        )

    # Clean categorical columns
    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df[col] = df[col].replace("", "Unknown")

    return df


def check_identical_rows(df: pd.DataFrame) -> bool:
    """
    Return True if all rows are identical.
    """
    if df.empty:
        return False
    return len(df.drop_duplicates()) == 1 and len(df) > 1


# =====================================================
# Template Generator
# =====================================================
def generate_batch_template(metadata: dict, num_rows: int = 5) -> pd.DataFrame:
    """
    Generate a blank template for batch upload.
    """
    feature_columns = metadata["feature_columns"]
    template_df = pd.DataFrame(columns=feature_columns)

    for _ in range(num_rows):
        template_df.loc[len(template_df)] = [""] * len(feature_columns)

    return template_df


# =====================================================
# Sidebar
# =====================================================
def render_sidebar(metadata: dict):
    """
    Render sidebar information.
    """
    with st.sidebar:
        st.header("Model Info")

        st.write(f"**Target:** {metadata.get('target', 'N/A')}")
        st.write(f"**Created At (UTC):** {metadata.get('created_at', 'N/A')}")

        metrics = metadata.get("metrics", {})
        if metrics:
            st.markdown("### Validation Metrics")
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")

        preprocess_summary = metadata.get("preprocess_summary", {})
        if preprocess_summary:
            st.markdown("### Training Data Cleanup")
            st.write(preprocess_summary)

        st.markdown("### Required Features")
        st.write(metadata.get("feature_columns", []))

        st.info(
            "Single Prediction now supports custom category values. "
            "If a value is not available in the dropdown, choose "
            f"**{CUSTOM_OPTION_LABEL}** and type it manually."
        )


# =====================================================
# Single Prediction
# =====================================================
def single_prediction_form(metadata: dict):
    """
    Render single prediction form and return validated DataFrame.
    Supports both dropdown selection and manual category input.
    """
    st.subheader("Single Prediction")

    numeric_defaults = metadata["ui_defaults"]["numeric_defaults"]
    categorical_options = metadata["ui_defaults"]["categorical_options"]
    feature_columns = metadata["feature_columns"]
    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["categorical_columns"]

    st.caption(
        "For category fields, you can either choose an existing value from the dropdown "
        "or select **Other / Enter manually** to type your own value."
    )

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}

        # -----------------------------------------------
        # Numeric Inputs
        # -----------------------------------------------
        with col1:
            st.markdown("### Numerical Inputs")
            for col in numeric_columns:
                default_val = float(numeric_defaults.get(col, 0.0))
                step = 1.0 if col == "weeknum" else 0.01

                input_data[col] = st.number_input(
                    label=col,
                    value=max(0.0, default_val),
                    min_value=0.0,
                    step=step,
                    help=f"Enter non-negative value for {col}"
                )

        # -----------------------------------------------
        # Categorical Inputs
        # -----------------------------------------------
        with col2:
            st.markdown("### Categorical Inputs")

            for col in categorical_columns:
                options = categorical_options.get(col, [])
                options = sorted([str(x) for x in options if str(x).strip() != ""])

                dropdown_options = options + [CUSTOM_OPTION_LABEL]

                selected_value = st.selectbox(
                    label=col,
                    options=dropdown_options,
                    key=f"{col}_select",
                    help=f"Select an existing value for {col} or choose '{CUSTOM_OPTION_LABEL}'"
                )

                if selected_value == CUSTOM_OPTION_LABEL:
                    manual_value = st.text_input(
                        label=f"Enter custom value for {col}",
                        key=f"{col}_manual",
                        placeholder=f"Type custom {col} value here"
                    ).strip()

                    input_data[col] = manual_value if manual_value else "Unknown"
                else:
                    input_data[col] = selected_value

        submitted = st.form_submit_button("Predict Profit")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df = validate_input_df(input_df, feature_columns, numeric_columns)
        return input_df

    return None


def render_single_prediction(model, metadata: dict):
    """
    Handle single prediction workflow.
    """
    try:
        input_df = single_prediction_form(metadata)

        if input_df is not None:
            raw_pred = model.predict(input_df)[0]

            # Safety layer: avoid showing negative prediction
            final_pred = max(0.0, raw_pred)

            st.success(f"Predicted Profit: **{final_pred:,.2f}**")

            with st.expander("Prediction Details"):
                st.write(f"Raw Model Prediction: **{raw_pred:,.2f}**")
                st.write(f"Displayed Prediction: **{final_pred:,.2f}**")
                st.markdown("### Input Used")
                st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        logger.exception("Single prediction failed")
        st.error(f"Single prediction failed: {e}")


# =====================================================
# Batch Prediction
# =====================================================
def render_batch_prediction(model, metadata: dict):
    """
    Handle batch prediction workflow.
    """
    st.subheader("Batch Prediction")

    feature_columns = metadata["feature_columns"]
    numeric_columns = metadata["numeric_columns"]

    st.markdown("### Step 1: Download Batch Template")
    template_df = generate_batch_template(metadata, num_rows=5)

    template_buffer = io.StringIO()
    template_df.to_csv(template_buffer, index=False)

    st.download_button(
        label="Download Batch Template CSV",
        data=template_buffer.getvalue(),
        file_name="batch_prediction_template.csv",
        mime="text/csv"
    )

    st.info(
        "Download the template, fill it with non-negative numeric values, "
        "and upload the same file below."
    )

    with st.expander("Required Columns for Batch Upload"):
        st.write(feature_columns)

    st.markdown("### Step 2: Upload Filled CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV for batch scoring",
        type=["csv"],
        key="batch_csv"
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded File Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            validated_df = validate_input_df(batch_df, feature_columns, numeric_columns)

            if check_identical_rows(validated_df):
                st.warning(
                    "All uploaded rows appear to be identical. Predictions may also be identical."
                )

            if st.button("Run Batch Prediction"):
                raw_preds = model.predict(validated_df)

                # Safety layer: prevent negative output display
                final_preds = np.maximum(raw_preds, 0)

                result_df = batch_df.copy()
                result_df["Raw_Predicted_Profit"] = raw_preds
                result_df["Predicted_Profit"] = final_preds

                st.success("Batch prediction completed successfully.")

                st.markdown("### Prediction Results Preview")
                st.dataframe(result_df.head(20), use_container_width=True)

                output_buffer = io.StringIO()
                result_df.to_csv(output_buffer, index=False)

                st.download_button(
                    label="Download Predictions CSV",
                    data=output_buffer.getvalue(),
                    file_name="batch_predictions_output.csv",
                    mime="text/csv"
                )

        except Exception as e:
            logger.exception("Batch prediction failed")
            st.error(f"Batch prediction failed: {e}")


# =====================================================
# Main
# =====================================================
def main():
    st.title("📈 Profit Prediction App")
    st.caption("Production-ready Streamlit app with single and batch prediction support")

    try:
        artifact = load_artifact(MODEL_PATH)
        model = artifact["model"]
        metadata = artifact["metadata"]

        render_sidebar(metadata)

        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

        with tab1:
            render_single_prediction(model, metadata)

        with tab2:
            render_batch_prediction(model, metadata)

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run the training script first:")
        st.code(
            "python train.py --input data/your_dataset.csv --output artifacts/profit_model.joblib",
            language="bash"
        )

    except Exception as e:
        logger.exception("Application failed")
        st.error(f"Application error: {e}")


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    main()