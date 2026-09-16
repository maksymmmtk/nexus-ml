import os
import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "nexus-ml-models")
MODELS_DIR = "./downloaded_models"
PROD_PREFIX = "production/"

os.makedirs(MODELS_DIR, exist_ok=True)

def get_s3_client():
    return boto3.client('s3', region_name=AWS_REGION)

def download_model_from_s3(model_name: str) -> tuple[str | None, str | None]:
    """Loads model and their manifest. Returns (model_path, manifest_path)"""
    s3 = get_s3_client()
    model_local_path = os.path.join(MODELS_DIR, model_name)
    manifest_name = model_name.replace(".onnx", ".json")
    manifest_local_path = os.path.join(MODELS_DIR, manifest_name)

    try:
        print(f"Loading {model_name} and {manifest_name} from S3 {PROD_PREFIX}...")
        s3.download_file(BUCKET_NAME, f"{PROD_PREFIX}{model_name}", model_local_path)
        s3.download_file(BUCKET_NAME, f"{PROD_PREFIX}{manifest_name}", manifest_local_path)
        return model_local_path, manifest_local_path
    except ClientError as e:
        print(f"AWS S3 error (download): {e}")
        return None, None