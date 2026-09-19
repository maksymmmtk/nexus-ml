import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from app.model_loader import download_model_from_s3
from app.ml_service import ml_service
from app.schemas import PredictRequest, PredictResponse
from app.cache import redis_client, get_cache_key

# Configure the logger for CloudWatch / AWS Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Fetch the target model name from environment variables
TARGET_MODEL = os.getenv("MODEL_NAME")

# Custom Prometheus metric to monitor Redis cache efficiency (Hit Rate)
cache_events = Counter(
    "nexusml_cache_events_total",
    "Number of cache hits and misses",
    ["model_family", "result"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: verify environment variables
    if not TARGET_MODEL:
        logger.critical("MODEL_NAME environment variable is missing!")
        raise RuntimeError("CRITICAL: MODEL_NAME environment variable is missing!")
        
    logger.info(f"Starting initialization for model: {TARGET_MODEL}")
    
    # Download the model weights and manifest from S3
    model_path, manifest_path = download_model_from_s3(TARGET_MODEL)
    
    if not model_path or not manifest_path:
        logger.critical(f"Failed to load {TARGET_MODEL} from S3 production/")
        raise RuntimeError(f"CRITICAL: Failed to load {TARGET_MODEL} from S3 production/")
        
    # Load the downloaded model into memory
    ml_service.load_model(TARGET_MODEL, model_path, manifest_path)
    logger.info(f"Model {TARGET_MODEL} loaded successfully into memory.")
    
    yield
    
    # Shutdown logic: clean up resources before the container stops
    logger.info("Shutting down container gracefully...")

# Initialize FastAPI application with the defined lifespan manager
app = FastAPI(title=f"NexusML API - {TARGET_MODEL}", lifespan=lifespan)

# Expose default Prometheus metrics (e.g., http_requests_total) automatically
Instrumentator().instrument(app).expose(app)


@app.get("/")
async def root():
    return {"message": f"NexusML API for {TARGET_MODEL} is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Generate a unique cache key based on the model and input features
    cache_key = get_cache_key(TARGET_MODEL, request.features)
    cached_result = None
    
    # 1. Redis cache check
    try:
        cached_result = redis_client.get(cache_key)
    except Exception as e:
        logger.warning(f"Redis is unavailable for GET. Reason: {e}")

    # Return cached result if available (Cache Hit)
    if cached_result:
        cache_events.labels(model_family=TARGET_MODEL, result="hit").inc()
        return PredictResponse(
            model_used=TARGET_MODEL,
            prediction=float(cached_result), 
            status="success (from cache)"
        )

    # 2. Inference computation (Cache Miss)
    try:
        prediction = ml_service.predict(request.features)
        cache_events.labels(model_family=TARGET_MODEL, result="miss").inc()
        
        # 3. Save the new prediction to Redis cache for 1 hour (3600 seconds)
        try:
            redis_client.setex(cache_key, 3600, prediction)
        except Exception as e:
            logger.warning(f"Redis is unavailable for SET. Reason: {e}")
            
        return PredictResponse(
            model_used=TARGET_MODEL,
            prediction=prediction, 
            status="success (computed)"
        )
    except ValueError as ve:
        # Handle invalid input data format
        logger.error(f"Validation error during prediction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Handle internal model crashes or unexpected errors
        logger.error(f"Internal model error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    # Verify Redis connection status via ping
    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
    except Exception:
        redis_status = "error"

    model_loaded = ml_service.is_loaded()

    # Determine overall health status (API is ready only if model is loaded and Redis is alive)
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded into memory"
        )

    is_degraded = redis_status != "connected"

    return {
        "status": "degraded" if is_degraded else "healthy",
        "model": TARGET_MODEL,
        "services": {
            "api": "connected",
            "redis": redis_status,
            "model_loaded": model_loaded
        }
    }