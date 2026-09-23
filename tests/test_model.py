import os
import json
import time
import pytest
import onnxruntime as ort
import numpy as np

MODELS_DIR = os.getenv("MODELS_DIR", "./scripts/staging")

def write_github_summary(model_name: str, exp_feat: int, act_feat: int, metrics: dict, latency: float, max_latency: float):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    status_schema = "Pass" if exp_feat == act_feat else f"Fail (Got {act_feat})"
    status_lat = "Pass" if latency <= max_latency else "Fail"

    # Creating a dynamic metrics table
    metrics_rows = ""
    for metric_name, (value, threshold, passed) in metrics.items():
        status_str = "Pass" if passed else f"Fail (Min: {threshold})"
        metrics_rows += f"| **{metric_name}** | {value:.4f} | {status_str} |\n"

    report = f"""### Test Report: `{model_name}`
| Metric | Value | Status |
|--------|-------|--------|
| **Schema Validation** | Expected: {exp_feat} | {status_schema} |
{metrics_rows}| **Avg Latency/Item** | {latency:.2f} ms | {status_lat} |

---
"""
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(report)


def get_model_test_cases():
    test_cases = []
    if not os.path.exists(MODELS_DIR):
        return test_cases

    for file in os.listdir(MODELS_DIR):
        if file.endswith(".onnx"):
            model_path = os.path.join(MODELS_DIR, file)
            manifest_path = os.path.join(MODELS_DIR, file.replace(".onnx", ".json"))

            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                test_cases.append(pytest.param(model_path, config, id=file))

    return test_cases

pytestmark = pytest.mark.skipif(
    not get_model_test_cases(),
    reason=f"No (.onnx + .json) pairs found in the folder {MODELS_DIR}"
)

@pytest.mark.parametrize("model_path, config", get_model_test_cases())
def test_model_validation_pipeline(model_path, config):
    # Import heavy libraries only within the test to avoid errors in app_pipeline
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    input_name = inputs[0].name

    actual_features = inputs[0].shape[1]
    expected_features = config["expected_features"]

    assert actual_features == expected_features, \
        f"{model_path}: Expected {expected_features} features, recieved {actual_features}"

    # 1. Loading dataset
    dataset_file = config.get("test_dataset")
    dataset_path = os.path.join(MODELS_DIR, dataset_file)
    assert os.path.exists(dataset_path), f"Dataset file {dataset_file} not found."

    df = pd.read_csv(dataset_path)
    target_col = config.get("target_column", "target")
    y_true = df[target_col].values
    X_test = df.drop(columns=[target_col]).values.astype(np.float32)

    # 2. Batch Inference and speed measurement
    start_time = time.time()
    result = session.run(None, {input_name: X_test})
    end_time = time.time()

    num_samples = len(X_test)
    avg_latency_ms = ((end_time - start_time) / num_samples) * 1000
    max_latency = float(config.get("max_latency_ms", 50.0))

    raw_predictions = result[0]
    task_type = config.get("task_type", "classification")
    metrics_result = {}
    is_metrics_pass = True

    # 3. Calculation of metrics
    if task_type == "classification":
        # Converting probabilities into classes
        if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
            y_pred = np.argmax(raw_predictions, axis=1) # Multiclass
        else:
            y_pred = (raw_predictions.flatten() > 0.5).astype(int) # Binary

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        min_f1 = float(config.get("min_f1_score", 0.80))
        is_metrics_pass = f1 >= min_f1

        metrics_result = {
            "Accuracy": (acc, None, True),
            "Precision": (prec, None, True),
            "Recall": (rec, None, True),
            "F1-Score": (f1, min_f1, is_metrics_pass)
        }
    elif task_type == "regression":
        y_pred = raw_predictions.flatten()
        r2 = r2_score(y_true, y_pred)

        # For regression we can specify min_r2_score in manifest (or accept the default of 0.5)
        min_r2 = float(config.get("min_r2_score", 0.50))
        is_metrics_pass = r2 >= min_r2

        metrics_result = {
            "R2-Score": (r2, min_r2, is_metrics_pass)
        }

    # 4. Report record
    model_name = os.path.basename(model_path)
    write_github_summary(
        model_name, expected_features, actual_features,
        metrics_result, avg_latency_ms, max_latency
    )

    # 5. Final checks
    assert is_metrics_pass, f"{model_path}: Validation metrics fell below the acceptable threshold."
    assert avg_latency_ms < max_latency, f"{model_path}: Inference {avg_latency_ms:.2f} ms exceeds the limit {max_latency} ms"