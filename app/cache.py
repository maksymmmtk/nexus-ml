import redis

redis_client = redis.Redis(
    host='redis', 
    port=6379, 
    db=0, 
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1
)

def get_cache_key(model_name: str, features: list[float]) -> str:
    return f"predict:{model_name}:{','.join(map(str, features))}"