import onnxruntime as ort
import numpy as np

class MLModelService:
    def __init__(self):
        self.session = None

    def load_model(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        print("ONNX Runtime session successfully initialized.")

    def predict(self, features: list):
        if not self.session:
            raise RuntimeError("Model not loaded")
        
        input_data = np.array([features], dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        
        result = self.session.run(None, {input_name: input_data})
        
        return int(result[0][0])
        
ml_service = MLModelService()