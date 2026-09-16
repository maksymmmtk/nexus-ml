import os
import json
import time
import pytest
import onnxruntime as ort
import numpy as np

MODELS_DIR = os.getenv("MODELS_DIR", "./scripts/staging")

def get_model_test_cases():
    test_cases = []
    if not os.path.exists(MODELS_DIR):
        return test_cases
        
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".onnx"):
            model_path = os.path.join(MODELS_DIR, file)
            manifest_path = os.path.join(MODELS_DIR, file.replace(".onnx", ".json"))
            
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    config = json.load(f)
                test_cases.append(pytest.param(model_path, config, id=file))
                
    return test_cases

pytestmark = pytest.mark.skipif(
    not get_model_test_cases(), 
    reason=f"No (.onnx + .json) pair found in the folder {MODELS_DIR}"
)

@pytest.mark.parametrize("model_path, config", get_model_test_cases())
def test_model_architecture_and_prediction(model_path, config):
    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    
    # 1. Shape Validation
    actual_features = inputs[0].shape[1]
    expected_features = config["expected_features"]
    assert actual_features == expected_features, \
        f"{model_path}: Expected {expected_features} features, recieved {actual_features}"
    
    # 2. Accuracy Check
    test_input = np.array([config["test_input"]], dtype=np.float32)
    input_name = inputs[0].name
    result = session.run(None, {input_name: test_input})
    
    predicted_val = float(result[0][0][0] if isinstance(result[0][0], (list, np.ndarray)) else result[0][0])
    expected_val = float(config["expected_class"])
    
    assert predicted_val == pytest.approx(expected_val, rel=1e-3), \
        f"{model_path}: Expected {expected_val}, recieved {predicted_val}"

@pytest.mark.parametrize("model_path, config", get_model_test_cases())
def test_model_latency(model_path, config):
    session = ort.InferenceSession(model_path)
    test_input = np.array([config["test_input"]], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    
    start_time = time.time()
    for _ in range(50):
        session.run(None, {input_name: test_input})
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / 50) * 1000
    assert avg_latency_ms < config["max_latency_ms"], \
        f"{model_path}: Inference {avg_latency_ms:.2f} ms exceeds the limit"