import os
import redis
import hashlib
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Fast timeouts prevent the API from hanging if Redis goes down
redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1
)

def get_cache_key(model_name: str, features: list[float]) -> str:
    # Hash the features array to keep the Redis key short and consistent in size
    features_str = json.dumps(features)
    features_hash = hashlib.sha256(features_str.encode("utf-8")).hexdigest()
    return f"predict:{model_name}:{features_hash}"