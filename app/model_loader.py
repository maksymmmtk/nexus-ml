import os
import boto3
from botocore.exceptions import ClientError

# Retrieving options from .env
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "nexus-ml-models")
MODELS_DIR = "./downloaded_models"

os.makedirs(MODELS_DIR, exist_ok=True)

def get_s3_client():
    # boto3 automatically picks up AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from the environment
    return boto3.client('s3', region_name=AWS_REGION)

def list_models_in_s3() -> list[str]:
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.onnx')]
        return []
    except ClientError as e:
        print(f"AWS S3 error (list): {e}")
        return []

def download_model_from_s3(model_name: str) -> str | None:
    local_path = f"{MODELS_DIR}/{model_name}"
    
    if os.path.exists(local_path):
        return local_path

    s3 = get_s3_client()
    try:
        print(f"Loading {model_name} from Amazon S3...")
        s3.download_file(BUCKET_NAME, model_name, local_path)
        return local_path
    except ClientError as e:
        print(f"AWS S3 error (download): {e}")
        return None