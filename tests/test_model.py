import os
import json
import time
import pytest
import onnxruntime as ort
import numpy as np

# У CI/CD сюди будуть завантажені файли зі staging-бакета
MODELS_DIR = os.getenv("MODELS_DIR", "./scripts/staging")

def get_model_test_cases():
    """Сканує папку, знаходить .onnx файли та їхні .json маніфести"""
    test_cases = []
    if not os.path.exists(MODELS_DIR):
        return test_cases
        
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".onnx"):
            model_path = os.path.join(MODELS_DIR, file)
            manifest_path = os.path.join(MODELS_DIR, file.replace(".onnx", ".json"))
            
            # Додаємо в тести тільки якщо у моделі є супутній маніфест
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    config = json.load(f)
                # Додаємо назву файлу для гарного виводу в консоль
                test_cases.append(pytest.param(model_path, config, id=file))
                
    return test_cases

pytestmark = pytest.mark.skipif(
    not get_model_test_cases(), 
    reason=f"Не знайдено жодної пари (.onnx + .json) у папці {MODELS_DIR}"
)

@pytest.mark.parametrize("model_path, config", get_model_test_cases())
def test_model_architecture_and_prediction(model_path, config):
    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    
    # 1. Shape Validation
    actual_features = inputs[0].shape[1]
    expected_features = config["expected_features"]
    assert actual_features == expected_features, \
        f"{model_path}: Очікувалось {expected_features} фічей, отримано {actual_features}"
    
    # 2. Accuracy Check
    test_input = np.array([config["test_input"]], dtype=np.float32)
    input_name = inputs[0].name
    result = session.run(None, {input_name: test_input})
    
    predicted_val = float(result[0][0][0] if isinstance(result[0][0], (list, np.ndarray)) else result[0][0])
    expected_val = float(config["expected_class"])
    
    assert predicted_val == pytest.approx(expected_val, rel=1e-3), \
        f"{model_path}: Очікувалося {expected_val}, отримано {predicted_val}"

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
        f"{model_path}: Інференс {avg_latency_ms:.2f} мс перевищує ліміт"