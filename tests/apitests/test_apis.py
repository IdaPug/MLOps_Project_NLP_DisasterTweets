from fastapi.testclient import TestClient

from project.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Disaster Tweet Classifier API"}


def test_predict_disaster():
    response = client.post(
        "/predict",
        json={"text": "There is a huge earthquake and people need help"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "disaster"
    assert 0 <= data["confidence"] <= 1


def test_predict_not_disaster():
    response = client.post(
        "/predict",
        json={"text": "I really enjoyed this movie last night"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["label"] == "not disaster"
    assert 0 <= data["confidence"] <= 1
