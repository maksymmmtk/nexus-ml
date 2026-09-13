import redis

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def get_cache_key(features: list[float]) -> str:
    return f"predict:{','.join(map(str, features))}"