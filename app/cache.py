import redis

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def get_cache_key(model_name: str, features: list[float]) -> str:
    # Example: predict:iris_v1.onnx:5.1,3.5,1.4,0.2
    return f"predict:{model_name}:{','.join(map(str, features))}"