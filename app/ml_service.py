import onnxruntime as ort
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self):
        self.session = None
        self.model_name = None
        self.manifest = None

    def load_model(self, model_name: str, model_path: str, manifest_path: str):
        self.model_name = model_name
        self.session = ort.InferenceSession(model_path)
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        logger.info(f"Model {model_name} successfully initialized for this container.")

    def is_loaded(self) -> bool:
        return self.session is not None

    def predict(self, features: list[float]) -> float:
        if not self.is_loaded():
            raise RuntimeError("Model not initialized.")
            
        # Feature count validation based on the loaded manifest metadata
        expected_features = self.manifest.get("expected_features")
        if expected_features and len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(features)}")

        # Convert input features to a 2D numpy array required by ONNX (batch_size=1)
        input_data = np.array([features], dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        
        # Execute the ONNX inference session
        result = self.session.run(None, {input_name: input_data})
        
        # Extract the scalar prediction value robustly
        prediction_value = result[0].ravel()[0]
        return float(prediction_value)

ml_service = InferenceService()