import onnxruntime as ort
import numpy as np
import json

class MLModelService:
    def __init__(self):
        self.session = None
        self.model_name = None
        self.manifest = None

    def load_model(self, model_name: str, model_path: str, manifest_path: str):
        self.model_name = model_name
        self.session = ort.InferenceSession(model_path)
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        print(f"Model {model_name} successfully initialized for this container.")

    def is_loaded(self) -> bool:
        return self.session is not None

    def predict(self, features: list):
        if not self.is_loaded():
            raise RuntimeError("Model not initialized.")
            
        # Validation of multiple features based on a manifest
        expected_features = self.manifest.get("expected_features")
        if len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(features)}")

        input_data = np.array([features], dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: input_data})
        
        # Universal return type (float works for both classification and regression)
        prediction_value = result[0][0][0] if isinstance(result[0][0], (list, np.ndarray)) else result[0][0]
        return float(prediction_value)

ml_service = MLModelService()