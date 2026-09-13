from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., 
        min_length=4, 
        max_length=4, 
        json_schema_extra={"example": [5.1, 3.5, 1.4, 0.2]}
    )

class PredictResponse(BaseModel):
    predicted_class: int
    status: str = "success"