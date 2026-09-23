import json
import os
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    fetch_california_housing,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define path to staging directory inside scripts
STAGING_DIR = os.path.join("scripts", "staging")
os.makedirs(STAGING_DIR, exist_ok=True)


def train_export_and_manifest(
    model, X, y, filename, feature_names=None, is_regression=False
):
    print(f"Training and exporting: {filename}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    # 1. Export .onnx
    initial_type = [(
        "float_input",
        FloatTensorType([None, X_train.shape[1]]),
    )]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    model_path = os.path.join(STAGING_DIR, filename)
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 2. Export Test Dataset (.csv)
    dataset_filename = filename.replace(".onnx", "_test.csv")
    dataset_path = os.path.join(STAGING_DIR, dataset_filename)
    
    # Creates DataFrame with test data
    columns = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X_test.shape[1])]
    df_test = pd.DataFrame(X_test, columns=columns)
    df_test["target"] = y_test
    df_test.to_csv(dataset_path, index=False)

    # 3. Generate individual .json manifest (Clean Config)
    manifest_data = {
        "expected_features": X_train.shape[1],
        "max_latency_ms": 100,
        "min_f1_score": 0.80 if not is_regression else None,
        "test_dataset": dataset_filename,
        "target_column": "target",
        "task_type": "regression" if is_regression else "classification"
    }

    manifest_filename = filename.replace(".onnx", ".json")
    manifest_path = os.path.join(STAGING_DIR, manifest_filename)

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=4)

    print(
        f"Artifacts saved in {STAGING_DIR}: {filename}, {manifest_filename}, {dataset_filename}\n"
    )


if __name__ == "__main__":
    print("Starting ML artifact creation...\n")
    iris = load_iris()
    train_export_and_manifest(
        RandomForestClassifier(n_estimators=100, random_state=42),
        iris.data,
        iris.target,
        "rf_iris_v1.onnx",
        feature_names=iris.feature_names
    )
    
    cancer = load_breast_cancer()
    train_export_and_manifest(
        LogisticRegression(max_iter=10000, random_state=42),
        cancer.data,
        cancer.target,
        "cancer_binary_model.onnx",
        feature_names=cancer.feature_names
    )
    
    california = fetch_california_housing()
    train_export_and_manifest(
        GradientBoostingRegressor(random_state=42),
        california.data,
        california.target,
        "california_regression_model.onnx",
        feature_names=california.feature_names,
        is_regression=True,
    )
    
    digits = load_digits()
    train_export_and_manifest(
        RandomForestClassifier(n_estimators=20, random_state=42),
        digits.data,
        digits.target,
        "digits_multiclass_model.onnx",
        feature_names=digits.feature_names
    )
    
    wine = load_wine()
    train_export_and_manifest(
        SVC(probability=True, random_state=42),
        wine.data,
        wine.target,
        "wine_svc_model.onnx",
        feature_names=wine.feature_names
    )