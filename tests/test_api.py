import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from botocore.exceptions import ClientError
from pydantic import ValidationError
from fastapi.testclient import TestClient


# 1. Environment Setup (MUST happen before importing main)
os.environ["MODEL_NAME"] = "test_model"
os.environ["AWS_DEFAULT_REGION"] = "eu-central-1"
os.environ["AWS_BUCKET_NAME"] = "test-bucket"

from app.schemas import PredictRequest
from app.cache import get_cache_key
from app.model_loader import download_model_from_s3
from app.ml_service import InferenceService
from main import app


# 2. Module Tests: Schemas
def test_predict_request_validation():
    # Verify that valid data passes
    req = PredictRequest(features=[1.0, 2.0, 3.0])
    assert req.features == [1.0, 2.0, 3.0]

    # Verify that empty lists are rejected (min_length=1)
    with pytest.raises(ValidationError):
        PredictRequest(features=[])


# 3. Module Tests: Cache Key Generation
def test_get_cache_key():
    model_name = "cancer_binary"
    features = [1.5, 2.5]
    key = get_cache_key(model_name, features)
    
    # Verify key structure and SHA-256 hash length
    assert key.startswith(f"predict:{model_name}:")
    hash_part = key.split(":")[-1]
    assert len(hash_part) == 64


# 4. Module Tests: AWS S3 Model Loader
@patch("app.model_loader.boto3.client")
def test_download_model_from_s3_success(mock_boto_client):
    # Setup mock S3 client
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3
    
    model_path, manifest_path = download_model_from_s3("test_model")
    
    # Verify paths and correct S3 bucket calls
    assert model_path == "./downloaded_models/test_model.onnx"
    assert manifest_path == "./downloaded_models/test_model.json"
    assert mock_s3.download_file.call_count == 2

@patch("app.model_loader.boto3.client")
def test_download_model_from_s3_failure(mock_boto_client):
    # Simulate an AWS ClientError (e.g., Access Denied or 404)
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Access Denied"}}, "GetObject"
    )
    mock_boto_client.return_value = mock_s3
    
    model_path, manifest_path = download_model_from_s3("missing_model")
    
    # Verify the loader handles the exception safely
    assert model_path is None
    assert manifest_path is None


# 5. Module Tests: ML Inference Service
@patch("app.ml_service.ort.InferenceSession")
def test_ml_service_load_and_predict(mock_ort_session):
    service = InferenceService()
    
    # Mock manifest data with expected features constraint
    manifest_data = json.dumps({"expected_features": 3})
    
    # Mock the ONNX session prediction output
    mock_session_instance = MagicMock()
    mock_session_instance.get_inputs.return_value = [MagicMock(name="input_tensor")]
    mock_session_instance.run.return_value = [np.array([[[0.95]]])] # Nested array output
    mock_ort_session.return_value = mock_session_instance

    with patch("builtins.open", mock_open(read_data=manifest_data)):
        service.load_model("test_model", "dummy.onnx", "dummy.json")
    
    assert service.is_loaded() is True

    # Test valid prediction
    result = service.predict([1.0, 2.0, 3.0])
    assert result == 0.95

    # Test feature count validation
    with pytest.raises(ValueError, match="Expected 3 features"):
        service.predict([1.0, 2.0])

def test_ml_service_uninitialized():
    service = InferenceService()
    with pytest.raises(RuntimeError, match="Model not initialized"):
        service.predict([1.0])


# 6. Integration Tests: FastAPI Application Routes
# Fixture to mock dependencies during the FastAPI lifespan and endpoints
@pytest.fixture
def mock_app_dependencies():
    with patch("main.download_model_from_s3") as mock_download, \
         patch("main.ml_service") as mock_ml, \
         patch("main.redis_client") as mock_redis:
         
        # Simulate successful model download and initialization
        mock_download.return_value = ("dummy.onnx", "dummy.json")
        mock_ml.is_loaded.return_value = True
        mock_ml.predict.return_value = 0.88
        mock_redis.ping.return_value = True
        
        yield mock_download, mock_ml, mock_redis

def test_root_endpoint(mock_app_dependencies):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "NexusML API for test_model is running"}

def test_predict_cache_hit(mock_app_dependencies):
    _, _, mock_redis = mock_app_dependencies
    # Simulate data found in Redis
    mock_redis.get.return_value = "0.75"

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.75
        assert data["status"] == "success (from cache)"
        mock_redis.setex.assert_not_called()

def test_predict_cache_miss_and_compute(mock_app_dependencies):
    _, mock_ml, mock_redis = mock_app_dependencies
    # Simulate empty cache
    mock_redis.get.return_value = None

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.88
        assert data["status"] == "success (computed)"
        
        # Verify inference was called and result was saved to Redis
        mock_ml.predict.assert_called_once_with([1.0, 2.0])
        mock_redis.setex.assert_called_once()

def test_predict_redis_down(mock_app_dependencies):
    _, mock_ml, mock_redis = mock_app_dependencies
    # Simulate Redis connection failure on GET and SET
    mock_redis.get.side_effect = Exception("Connection refused")
    mock_redis.setex.side_effect = Exception("Connection refused")

    with TestClient(app) as client:
        # API should survive Redis crash and compute prediction normally
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.88
        assert data["status"] == "success (computed)"
        mock_ml.predict.assert_called_once()

def test_predict_validation_error(mock_app_dependencies):
    _, mock_ml, _ = mock_app_dependencies
    # Simulate ML service rejecting feature count
    mock_ml.predict.side_effect = ValueError("Expected 3 features, got 2")

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        
        assert response.status_code == 400
        assert "Expected 3 features" in response.json()["detail"]

def test_predict_internal_error(mock_app_dependencies):
    _, mock_ml, _ = mock_app_dependencies
    # Simulate internal ONNX crash
    mock_ml.predict.side_effect = Exception("ONNX Core Fault")

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        
        assert response.status_code == 500
        assert "ONNX Core Fault" in response.json()["detail"]

def test_health_check_healthy(mock_app_dependencies):
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_health_check_degraded_redis(mock_app_dependencies):
    _, _, mock_redis = mock_app_dependencies
    # Simulate Redis ping failure
    mock_redis.ping.side_effect = Exception("Timeout")

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["redis"] == "error"

def test_health_check_model_not_loaded(mock_app_dependencies):
    _, mock_ml, _ = mock_app_dependencies
    # Simulate ML service failure to load
    mock_ml.is_loaded.return_value = False

    with TestClient(app) as client:
        response = client.get("/health")
        # Should return 503 Service Unavailable for load balancers
        assert response.status_code == 503
        assert response.json()["detail"] == "Model is not loaded into memory"


# 7. Lifespan Startup Failure Tests
def test_lifespan_s3_failure():
    # Temporarily remove mocked dependencies to test raw startup behavior
    with patch("main.download_model_from_s3") as mock_download:
        mock_download.return_value = (None, None)
        
        # TestClient triggers lifespan automatically on init
        with pytest.raises(RuntimeError, match="CRITICAL: Failed to load"):
            with TestClient(app):
                pass