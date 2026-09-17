import os
import sys
import boto3
import zipfile
from botocore.exceptions import NoCredentialsError

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
LOCAL_STAGING_DIR = os.path.join("scripts", "staging")

STAGING_PREFIX = "staging/"
PROD_PREFIX = "production/"

s3 = boto3.client('s3', region_name=REGION)

def upload_staging():
    print(f"Uploading local models to S3 ({STAGING_PREFIX})...")
    if not os.path.exists(LOCAL_STAGING_DIR):
        print("Warning: Local staging directory is empty.")
        return

    uploaded = 0
    for filename in os.listdir(LOCAL_STAGING_DIR):
        if filename.endswith('.zip'):
            local_path = os.path.join(LOCAL_STAGING_DIR, filename)
            s3_key = f"{STAGING_PREFIX}{filename}"
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            print(f"  - Uploaded archive to staging: {s3_key}")
            uploaded += 1
    print(f"Successfully uploaded {uploaded} archives.\n")

def download_staging():
    print(f"CI/CD Stage 1: Downloading model archives (.zip) from S3 ({STAGING_PREFIX})...")
    os.makedirs(LOCAL_STAGING_DIR, exist_ok=True)
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=STAGING_PREFIX)
    if 'Contents' not in response:
        print("Warning: The staging/ prefix in S3 is empty or does not exist.")
        return

    downloaded = 0
    for obj in response['Contents']:
        file_key = obj['Key']
        if file_key.endswith('/') or not file_key.endswith('.zip'):
            continue
        
        filename = os.path.basename(file_key)
        local_zip_path = os.path.join(LOCAL_STAGING_DIR, filename)
        
        s3.download_file(BUCKET_NAME, file_key, local_zip_path)
        print(f"  - Downloaded archive: {filename}")
        
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_STAGING_DIR)
        print(f"  - Extracted: {filename}")
        
        os.remove(local_zip_path)
        downloaded += 1
        
    print(f"Successfully downloaded and extracted {downloaded} archives.\n")

def promote_to_production():
    print(f"CI/CD Stage 2: Promoting validated models to production ({PROD_PREFIX}) and cleaning up staging...")
    if not os.path.exists(LOCAL_STAGING_DIR):
        print("Warning: Local staging directory is empty. Nothing to promote.")
        return

    promoted = 0
    for filename in os.listdir(LOCAL_STAGING_DIR):
        if filename.endswith('.onnx') or filename.endswith('.json'):
            local_path = os.path.join(LOCAL_STAGING_DIR, filename)
            s3_prod_key = f"{PROD_PREFIX}{filename}"
            
            s3.upload_file(local_path, BUCKET_NAME, s3_prod_key)
            print(f"  - Deployed to prod: {s3_prod_key}")
            promoted += 1
            
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=STAGING_PREFIX)
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.zip'):
                s3.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                print(f"  - Cleaned up from staging: {obj['Key']}")
            
    print(f"Successfully promoted {promoted} files and cleaned up staging archives.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python s3_sync.py [upload_staging|download|promote]")
        sys.exit(1)
        
    if not BUCKET_NAME:
        print("Error: AWS_BUCKET_NAME environment variable is not set!")
        sys.exit(1)
        
    action = sys.argv[1]
    try:
        if action == "upload_staging":
            upload_staging()
        elif action == "download":
            download_staging()
        elif action == "promote":
            promote_to_production()
        else:
            print(f"Error: Unknown action: {action}")
    except NoCredentialsError:
        print("Error: AWS credentials not found (check your .env file)!")