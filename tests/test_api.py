import os
import pytest
from unittest.mock import patch

os.environ["MODEL_NAME"] = "iris_model.onnx"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "is running" in response.json()["message"]

@patch("app.main.redis_client.get")
@patch("app.main.redis_client.setex")
@patch("app.main.ml_service.predict")
def test_predict_cache_miss(mock_predict, mock_setex, mock_get):
    mock_get.return_value = None
    mock_predict.return_value = 1.0

    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1.0
    assert "computed" in data["status"]
    assert data["model_used"] == "iris_model.onnx"

    mock_predict.assert_called_once_with([5.1, 3.5, 1.4, 0.2])
    mock_setex.assert_called_once()

@patch("app.main.redis_client.get")
@patch("app.main.ml_service.predict")
def test_predict_cache_hit(mock_predict, mock_get):
    mock_get.return_value = "0.0"

    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0.0
    assert "from cache" in data["status"]

    mock_predict.assert_not_called()

def test_predict_invalid_data():
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422

    response = client.post("/predict", json={"wrong_key": [1, 2, 3]})
    assert response.status_code == 422