from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from app.model_loader import download_model_from_s3, LOCAL_MODEL_PATH
from app.ml_service import ml_service
from app.schemas import PredictRequest, PredictResponse
from app.cache import redis_client, get_cache_key
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model from MinIO...")
    if download_model_from_s3():
        ml_service.load_model(LOCAL_MODEL_PATH)
    else:
        print("Error: Model not loaded.")
    yield
    if os.path.exists(LOCAL_MODEL_PATH):
        os.remove(LOCAL_MODEL_PATH)

app = FastAPI(title="NexusML API", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "NexusML API is running"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        cache_key = get_cache_key(request.features)
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            print(f"Cache Hit! Key: {cache_key}")
            return PredictResponse(
                predicted_class=int(cached_result), 
                status="success (from cache)"
            )
            
        print("Cache Miss! Running inference...")
        prediction = ml_service.predict(request.features)
        
        redis_client.setex(cache_key, 3600, prediction)
        
        return PredictResponse(
            predicted_class=prediction, 
            status="success (computed)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))