from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from app.model_loader import download_model_from_s3
from app.ml_service import ml_service
from app.schemas import PredictRequest, PredictResponse
from app.cache import redis_client, get_cache_key
from app.model_loader import list_models_in_s3


app = FastAPI(title="NexusML API")
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    return {"message": "NexusML API is running"}


@app.get("/models")
async def get_available_models():
    models = list_models_in_s3()
    return {"available_models": models, "total_count": len(models)}


@app.get("/models/active")
async def get_active_models():
    active = ml_service.get_loaded_models()
    return {"active_models": active, "total_count": len(active)}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model_name = request.model_name
    
    # 1. Redis cache check
    cache_key = get_cache_key(model_name, request.features)
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return PredictResponse(
            model_used=model_name,
            predicted_class=int(cached_result), 
            status="success (from cache)"
        )
        
    # 2. Checking for the presence of the model in RAM
    if not ml_service.is_loaded(model_name):
        # 3. "On-the-fly" loading from MinIO (Lazy Loading)
        local_path = download_model_from_s3(model_name)
        if not local_path:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in MinIO")
        ml_service.load_model(model_name, local_path)

    # 4. Inference
    try:
        prediction = ml_service.predict(model_name, request.features)
        redis_client.setex(cache_key, 3600, prediction)
        return PredictResponse(
            model_used=model_name,
            predicted_class=prediction, 
            status="success (computed)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/unload")
async def unload_model(model_name: str):
    if ml_service.unload_model(model_name):
        return {"message": f"Model {model_name} successfully unloaded from RAM"}
    raise HTTPException(status_code=404, detail="Model not found in memory")


@app.delete("/cache/clear")
async def clear_redis_cache():
    try:
        redis_client.flushdb()
        return {"message": "Redis cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {e}")


@app.get("/health")
async def deep_health_check():
    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
    except:
        redis_status = "error"
   
    return {
        "status": "healthy" if redis_status == "connected" else "degraded",
        "services": {
            "api": "connected",
            "redis": redis_status
        }
    }