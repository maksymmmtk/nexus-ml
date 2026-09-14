from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Model name (e.g., iris_v1.onnx)")
    features: list[float] = Field(..., min_length=1)

class PredictResponse(BaseModel):
    model_used: str
    predicted_class: int
    status: str = "success"