import os
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME = "models"
MODELS_DIR = "./downloaded_models"

os.makedirs(MODELS_DIR, exist_ok=True)

def list_models_in_s3() -> list[str]:
    try:
        client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        if client.bucket_exists(BUCKET_NAME):
            objects = client.list_objects(BUCKET_NAME)
            return [obj.object_name for obj in objects if obj.object_name.endswith('.onnx')]
        return []
    except S3Error as e:
        print(f"MinIO Error: {e}")
        return []

def download_model_from_s3(model_name: str) -> str | None:
    local_path = f"{MODELS_DIR}/{model_name}"
    
    # If the model is already downloaded locally, don't fetch it again.
    if os.path.exists(local_path):
        return local_path

    try:
        client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        if not client.bucket_exists(BUCKET_NAME):
            return None
            
        print(f"Loading {model_name} from MinIO...")
        client.fget_object(BUCKET_NAME, model_name, local_path)
        return local_path
    except S3Error as e:
        print(f"MinIO Error: {e}")
        return None