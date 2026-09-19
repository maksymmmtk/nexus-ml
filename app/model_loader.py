import os
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "nexus-ml-models")
MODELS_DIR = "./downloaded_models"
PROD_PREFIX = "production/"

os.makedirs(MODELS_DIR, exist_ok=True)

def get_s3_client():
    return boto3.client('s3', region_name=AWS_REGION)

def download_model_from_s3(model_base_name: str) -> tuple[str | None, str | None]:
    """
    Downloads the model and its manifest from S3. 
    Assumes model_base_name does not include file extensions.
    Returns (model_path, manifest_path)
    """
    s3 = get_s3_client()
    
    # Explicitly construct file names with extensions
    model_file = f"{model_base_name}.onnx"
    manifest_file = f"{model_base_name}.json"
    
    model_local_path = os.path.join(MODELS_DIR, model_file)
    manifest_local_path = os.path.join(MODELS_DIR, manifest_file)

    try:
        logger.info(f"Downloading {model_file} and {manifest_file} from S3 {PROD_PREFIX}...")
        s3.download_file(BUCKET_NAME, f"{PROD_PREFIX}{model_file}", model_local_path)
        s3.download_file(BUCKET_NAME, f"{PROD_PREFIX}{manifest_file}", manifest_local_path)
        return model_local_path, manifest_local_path
    except ClientError as e:
        logger.error(f"AWS S3 download error for model {model_base_name}: {e}")
        return None, None