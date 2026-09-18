import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1
)

def get_cache_key(model_name: str, features: list[float]) -> str:
    return f"predict:{model_name}:{','.join(map(str, features))}"