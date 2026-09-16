import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from app.model_loader import download_model_from_s3
from app.ml_service import ml_service
from app.schemas import PredictRequest, PredictResponse
from app.cache import redis_client, get_cache_key

TARGET_MODEL = os.getenv("MODEL_NAME")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if not TARGET_MODEL:
        raise RuntimeError("CRITICAL: MODEL_NAME environment variable is missing!")
        
    model_path, manifest_path = download_model_from_s3(TARGET_MODEL)
    if not model_path or not manifest_path:
        raise RuntimeError(f"CRITICAL: Failed to load {TARGET_MODEL} from S3 production/")
        
    ml_service.load_model(TARGET_MODEL, model_path, manifest_path)
    yield
    # Shutdown logic
    print("Shutting down container...")

app = FastAPI(title=f"NexusML API - {TARGET_MODEL}", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    return {"message": f"NexusML API for {TARGET_MODEL} is running"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    cache_key = get_cache_key(TARGET_MODEL, request.features)
    cached_result = None
    
    # 1. Redis cache check
    try:
        cached_result = redis_client.get(cache_key)
    except Exception as e:
        print(f"Redis is unavailable for GET: {e}")

    if cached_result:
        return PredictResponse(
            model_used=TARGET_MODEL,
            prediction=float(cached_result), 
            status="success (from cache)"
        )

    # 2. Inference
    try:
        prediction = ml_service.predict(request.features)
        
        # 3. Redis cache save
        try:
            redis_client.setex(cache_key, 3600, prediction)
        except Exception as e:
            print(f"Redis is unavailable for SET: {e}")
            
        return PredictResponse(
            model_used=TARGET_MODEL,
            prediction=prediction, 
            status="success (computed)"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def deep_health_check():
    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
    except:
        redis_status = "error"
   
    return {
        "status": "healthy" if redis_status == "connected" and ml_service.is_loaded() else "degraded",
        "model": TARGET_MODEL,
        "services": {
            "api": "connected",
            "redis": redis_status,
            "model_loaded": ml_service.is_loaded()
        }
    }