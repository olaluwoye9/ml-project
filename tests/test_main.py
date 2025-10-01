# tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World"}

def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_list_models():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_prediction_invalid_model():
    r = client.post("/predict/does_not_exist", json={"features": {"x": 1}})
    assert r.status_code == 404
    assert r.json()["detail"] == "Model not found"

def test_prediction_valid_model_mocks_model(mocker):
    # patch the function the endpoint calls
    # adjust the target string if your function lives elsewhere
    mock = mocker.patch("app.main.run_prediction", return_value={"model": "modelA", "prediction": "mocked"})
    r = client.post("/predict/modelA", json={"features": {"x": 1}})
    assert r.status_code == 200
    assert r.json()["prediction"] == "mocked"
    mock.assert_called_once_with("modelA", {"x": 1})
# ci trigger
