import os
import json
import time
import pytest
import onnxruntime as ort
import numpy as np

MODELS_DIR = os.getenv("MODELS_DIR", "./scripts/staging")

def write_github_summary(model_name: str, expected_feat: int, actual_feat: int, acc_pass: bool, latency: float, max_latency: float):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    status_schema = "Pass" if expected_feat == actual_feat else f"Fail (Got {actual_feat})"
    status_acc = "Pass" if acc_pass else "Fail"
    status_lat = "Pass" if latency <= max_latency else f"Fail"

    report = f"""### Test Report: `{model_name}`
| Metric | Result | Status |
|--------|--------|--------|
| **Schema Validation** | Expected: {expected_feat} | {status_schema} |
| **Prediction Match** | Tolerance: 1e-3 | {status_acc} |
| **Avg Latency (50 runs)** | {latency:.2f} ms | {status_lat} |

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
    reason=f"No (.onnx + .json) pair found in the folder {MODELS_DIR}"
)

@pytest.mark.parametrize("model_path, config", get_model_test_cases())
def test_model_validation_pipeline(model_path, config):
    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    input_name = inputs[0].name
    
    actual_features = inputs[0].shape[1]
    expected_features = config["expected_features"]
    
    test_input = np.array([config["test_input"]], dtype=np.float32)
    result = session.run(None, {input_name: test_input})
    
    predicted_val = float(result[0][0][0] if isinstance(result[0][0], (list, np.ndarray)) else result[0][0])
    expected_val = float(config["expected_class"])
    is_acc_pass = pytest.approx(expected_val, rel=1e-3) == predicted_val
    
    start_time = time.time()
    for _ in range(50):
        session.run(None, {input_name: test_input})
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / 50) * 1000
    max_latency = float(config.get("max_latency_ms", 50.0))

    model_name = os.path.basename(model_path)
    write_github_summary(
        model_name, 
        expected_features, 
        actual_features, 
        is_acc_pass, 
        avg_latency_ms, 
        max_latency
    )
    
    assert actual_features == expected_features, \
        f"{model_path}: Expected {expected_features} features, recieved {actual_features}"
        
    assert is_acc_pass, \
        f"{model_path}: Expected {expected_val}, recieved {predicted_val}"
        
    assert avg_latency_ms < max_latency, \
        f"{model_path}: Inference {avg_latency_ms:.2f} ms exceeds the limit {max_latency} ms"