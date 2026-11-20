"""
preprocessing.py
----------------
Everything related to loading, checking, cleaning and encoding the data.

Key idea (data-leakage prevention):
    The ColumnTransformer built here is placed INSIDE a scikit-learn Pipeline
    together with the model. When we call pipeline.fit(X_train, y_train) the
    scalers / encoders learn their parameters (mean, std, categories) ONLY from
    the training data. The test data is only transformed, never used for fitting.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------------
ID_COL = "customerID"      # identifier only - NEVER used as a feature
TARGET = "Churn"           # target variable (Yes / No)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]   # will be scaled
BINARY_FEATURES = ["SeniorCitizen"]                               # already 0/1
CATEGORICAL_FEATURES = [                                          # will be one-hot encoded
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


# ----------------------------------------------------------------------------
# Loading and inspection
# ----------------------------------------------------------------------------
def load_data(csv_path) -> pd.DataFrame:
    """Read the CSV file. Raises a clear error if the file is missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Keep 'telco_customer_churn_synthetic.csv' "
            "in the project folder (next to app.py)."
        )
    return pd.read_csv(csv_path)


def inspect_data(df: pd.DataFrame) -> dict:
    """Return a dictionary with shape, dtypes, missing values, duplicates and class balance."""
    counts = df[TARGET].value_counts()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COL].duplicated().sum()),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percent": {k: round(float(v) * 100, 2) for k, v in (counts / counts.sum()).items()},
    }


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_data(df: pd.DataFrame):
    """
    Clean the raw data and return (clean_df, report).

    Steps: strip stray spaces, make TotalCharges numeric, remove duplicate rows /
    duplicate customer IDs. Remaining missing values (if any) are handled by
    imputers inside the model pipeline so that they are learned from training data only.
    """
    df = df.copy()
    report = {"rows_before": len(df)}

    # 1. Remove accidental spaces in text columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype("object")

    # 2. TotalCharges must be numeric (blank strings become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # A brand-new customer (tenure = 0) has not been billed yet -> 0
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # 3. Remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=ID_COL, keep="first")

    # 4. Target must only contain Yes / No
    df = df[df[TARGET].isin(["Yes", "No"])].reset_index(drop=True)

    report["rows_after"] = len(df)
    report["rows_removed"] = report["rows_before"] - report["rows_after"]
    report["missing_after_cleaning"] = int(df[FEATURE_COLUMNS].isna().sum().sum())
    return df, report


# ----------------------------------------------------------------------------
# Encoding + scaling pipeline
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that:
      * numeric columns     -> median imputation + StandardScaler
      * binary column       -> passed through unchanged (already 0 / 1)
      * categorical columns -> most-frequent imputation + OneHotEncoder
    sparse_threshold=0 makes the output a dense array (needed by Naive Bayes).
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split. customerID is dropped here (not a feature).
    Every model uses this SAME split, so the comparison is fair.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET] == "Yes").astype(int)          # Yes -> 1 (churn), No -> 0
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
