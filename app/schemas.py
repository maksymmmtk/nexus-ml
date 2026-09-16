from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)

class PredictResponse(BaseModel):
    model_used: str
    prediction: float
    status: str = "success"