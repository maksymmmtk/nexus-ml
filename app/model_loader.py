import os
from minio import Minio
from minio.error import S3Error

MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME = "models"
MODEL_FILENAME = "iris_model.onnx"
LOCAL_MODEL_PATH = f"./{MODEL_FILENAME}"

def download_model_from_s3():
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        if not client.bucket_exists(BUCKET_NAME):
            print(f"Error: Bucket '{BUCKET_NAME}' not found.")
            return False
            
        print(f"Loading {MODEL_FILENAME} from MinIO...")
        client.fget_object(BUCKET_NAME, MODEL_FILENAME, LOCAL_MODEL_PATH)
        print("Model successfully loaded!")
        return True
        
    except S3Error as e:
        print(f"MinIO Error: {e}")
        return False