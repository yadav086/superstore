# Profit Prediction Training Pipeline

An optimized machine learning production pipeline that handles data validation, advanced engineering, feature selection, and training to predict **Profit**. This pipeline strips metadata overhead and applies high-ratio `lzma` compression to guarantee a tiny model file footprint, making it perfectly tailored for fast deployment in Streamlit applications.

---

## 🚀 Key Features

* **Data Validation & Cleaning**: Automatic schema validation, robust missing-value handling, and strict filter rules dropping rows containing invalid or negative financial values.
* **Dual-Type Preprocessing**: 
  * **Numerical**: Uses a `Median Imputer` combined with a `Yeo-Johnson PowerTransformer` to normalize metric skewness.
  * **Categorical**: Robust high-cardinality processing using `CatBoostEncoder` coupled with variance stabilization.
* **Cross-Validated Feature Selection**: Integrates recursive feature elimination (`RFECV`) driven by an internal `XGBRFRegressor` to dynamically identify the 8 most critical prediction variables.
* **Streamlit Application Metadata Bundle**: Generates a runtime payload tracking global metrics, cleaning histories, and pre-calculated UI defaults (medians and dropdown categories) to directly fuel interactive frontends.
* **Ultra-Lightweight Storage**: Leverages advanced structural stripping alongside a maximum-level `lzma` serialization stream to shrink the output pickle file down to minimum size.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then execute the following command to download the required package libraries:
   ```bash
   pip install numpy pandas scikit-learn category_encoders xgboost joblib
   ```

---

## 📦 Expected Data Schema

Your input data file (`.csv` or `.parquet`) must contain the following structural fields:


| Data Type | Column Name |
| :--- | :--- |
| **Target Variable** | `Profit` |
| **Numerical Features** | `Discount`, `Sales`, `Shipping Cost`, `weeknum` |
| **Categorical Features** | `Category`, `City`, `Country`, `Order ID`, `Order Priority`, `Product ID`, `Product Name`, `Sub-Category` |

---

## 💻 How To Run the Pipeline

The framework utilizes an argument parser engine. Run the script directly from your terminal by passing your input data path and desired artifact storage path:

```bash
python train.py --input_path "path/to/your/dataset.csv" --output_path "./outputs/model_artifact.pkl"
```

### Command Line Options:
* `--input_path` *(Required)*: Path to your raw input dataset (`.csv` or `.parquet`).
* `--output_path` *(Optional)*: Target location to save the compressed model bundle. Defaults to `./outputs/model_artifact.pkl`.

---

## 📤 Output Artifact Structure

The script outputs a single optimized compressed file containing a dictionary with two primary production modules:

```python
{
    "model": Pipeline(...),     # Complete Preprocessing + Feature Selection + RandomForest Regressor
    "metadata": {               # Metadata dictionary for frontend rendering
        "created_at": "...",
        "target": "Profit",
        "metrics": {"r2": ..., "mae": ..., "rmse": ...},
        "ui_defaults": {
            "numeric_defaults": {...},     # Pre-calculated feature medians
            "categorical_options": {...}   # Sorted unique values list for dropdown menus
        }
    }
}
```
