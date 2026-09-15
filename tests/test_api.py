import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "NexusML API is running"}

@patch("app.main.redis_client.get")
@patch("app.main.redis_client.setex")
@patch("app.main.ml_service.is_loaded")
@patch("app.main.download_model_from_s3")
@patch("app.main.ml_service.load_model")
@patch("app.main.ml_service.predict")
def test_predict_cache_miss(mock_predict, mock_load, mock_download, mock_is_loaded, mock_setex, mock_get):
    mock_get.return_value = None
    mock_is_loaded.return_value = False
    mock_download.return_value = "./downloaded_models/iris_model.onnx"
    mock_predict.return_value = 1

    payload = {
        "model_name": "iris_model.onnx",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert "computed" in data["status"]
    assert data["model_used"] == "iris_model.onnx"

    mock_is_loaded.assert_called_once_with("iris_model.onnx")
    mock_download.assert_called_once_with("iris_model.onnx")
    mock_load.assert_called_once_with("iris_model.onnx", "./downloaded_models/iris_model.onnx")
    mock_predict.assert_called_once_with("iris_model.onnx", [5.1, 3.5, 1.4, 0.2])
    mock_setex.assert_called_once()

@patch("app.main.redis_client.get")
@patch("app.main.ml_service.predict")
def test_predict_cache_hit(mock_predict, mock_get):
    mock_get.return_value = "0"

    payload = {
        "model_name": "iris_model.onnx",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 0
    assert "from cache" in data["status"]

    mock_predict.assert_not_called()

def test_predict_invalid_data():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 422

    response = client.post("/predict", json={
        "model_name": "iris_model.onnx", 
        "features": []
    })
    assert response.status_code == 422