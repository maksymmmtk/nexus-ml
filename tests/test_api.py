import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

@pytest.fixture(autouse=True)
def mock_lifespan_s3():
    with patch("app.main.download_model_from_s3", return_value=True):
        yield

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "NexusML API is running"}

@patch("app.main.redis_client.get")
@patch("app.main.redis_client.setex")
@patch("app.main.ml_service.predict")
def test_predict_cache_miss(mock_predict, mock_setex, mock_get):
    mock_get.return_value = None
    mock_predict.return_value = 1

    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert "computed" in data["status"]
    
    mock_predict.assert_called_once()
    mock_setex.assert_called_once()

@patch("app.main.redis_client.get")
@patch("app.main.ml_service.predict")
def test_predict_cache_hit(mock_predict, mock_get):
    mock_get.return_value = "0"

    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 0
    assert "from cache" in data["status"]
    
    mock_predict.assert_not_called()

def test_predict_invalid_data():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4]})
    assert response.status_code == 422