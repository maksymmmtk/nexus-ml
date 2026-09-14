import onnxruntime as ort
import numpy as np

class MLModelService:
    def __init__(self):
        self.session = {} # Dictionary: {"iris_v1.onnx": session_object, "diabetes.onnx": session_object}

    def get_loaded_models(self) -> list[str]:
        return list(self.sessions.keys())

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self.sessions

    def load_model(self, model_name: str, model_path: str):
        if not self.is_loaded(model_name):
            self.sessions[model_name] = ort.InferenceSession(model_path)
            print("ONNX Runtime session successfully initialized.")

    def unload_model(self, model_name: str) -> bool:
        if model_name in self.sessions:
            del self.sessions[model_name]
            print(f"Model {model_name} unloaded form memory.")
            return True
        return False

    def predict(self, model_name: str, features: list):
        session = self.sessions.get(model_name)
        if not session:
            raise RuntimeError(f"Model {model_name} not initialized.")

        input_data = np.array([features], dtype=np.float32)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})
        return int(result[0][0])

ml_service = MLModelService()