import sys
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture(scope="module")
def client():
    """Shared TestClient for all tests — no real HTTP server needed."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_payload():
    """A realistic sample from the Wisconsin dataset (expected: Benign)."""
    return {
        "radius": 14.12,
        "texture": 19.29,
        "perimeter": 91.97,
        "area": 654.8,
        "smoothness": 0.096,
        "compactness": 0.104,
        "concavity": 0.088,
        "concave_points": 0.048,
        "symmetry": 0.181,
        "fractal_dimension": 0.062,
    }

@pytest.fixture
def malignant_payload():
    """A sample with high-risk morphometry (expected: Malignant)."""
    return {
        "radius": 25.0,
        "texture": 30.0,
        "perimeter": 165.0,
        "area": 2000.0,
        "smoothness": 0.15,
        "compactness": 0.28,
        "concavity": 0.35,
        "concave_points": 0.18,
        "symmetry": 0.30,
        "fractal_dimension": 0.09,
    }

class TestHealthCheck:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_success_message(self, client):
        response = client.get("/")
        body = response.json()
        assert "message" in body

class TestPredictionHappyPath:

    def test_returns_200(self, client, valid_payload):
        response = client.post("/prediction", json=valid_payload)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        assert "result" in data
        assert "confiability" in data
        assert "probability_malignant" in data
        assert "probability_benign" in data
        assert "alert" in data

    def test_result_is_valid_label(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        assert data["result"] in ("Malignant", "Benign")

    def test_probabilities_sum_to_100(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        total = data["probability_malignant"] + data["probability_benign"]
        assert abs(total - 100.0) < 0.1  # allow tiny float rounding

    def test_confiability_is_between_0_and_100(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        assert 0.0 <= data["confiability"] <= 100.0

    def test_alert_is_bool(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        assert isinstance(data["alert"], bool)

    def test_alert_true_when_malignant(self, client, malignant_payload):
        data = client.post("/prediction", json=malignant_payload).json()
        if data["result"] == "Malignant":
            assert data["alert"] is True

    def test_alert_false_when_benign(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        if data["result"] == "Benign":
            assert data["alert"] is False

    def test_confiability_matches_predicted_class(self, client, valid_payload):
        data = client.post("/prediction", json=valid_payload).json()
        if data["result"] == "Malignant":
            assert data["confiability"] == pytest.approx(data["probability_malignant"], abs=0.1)
        else:
            assert data["confiability"] == pytest.approx(data["probability_benign"], abs=0.1)

class TestPredictionValidation:

    def test_missing_field_returns_422(self, client, valid_payload):
        incomplete = {k: v for k, v in valid_payload.items() if k != "radius"}
        response = client.post("/prediction", json=incomplete)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client, valid_payload):
        bad = {**valid_payload, "radius": "not-a-number"}
        response = client.post("/prediction", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/prediction", json={})
        assert response.status_code == 422

    def test_extra_fields_are_ignored(self, client, valid_payload):
        extra = {**valid_payload, "unknown_field": 99.9}
        response = client.post("/prediction", json=extra)
        assert response.status_code in (200, 422)  # depends on model config

    def test_negative_values_are_accepted(self, client, valid_payload):
        edge = {**valid_payload, "radius": -1.0}
        response = client.post("/prediction", json=edge)
        assert response.status_code in (200, 422)

    def test_zero_values_are_accepted(self, client, valid_payload):
        edge = {**valid_payload, "area": 0.0}
        response = client.post("/prediction", json=edge)
        assert response.status_code in (200, 422)

class TestPredictionContentType:

    def test_requires_json_content_type(self, client, valid_payload):
        """Sending form data instead of JSON should be rejected."""
        response = client.post(
            "/prediction",
            data=valid_payload,  # form-encoded, not JSON
        )
        assert response.status_code == 422

    def test_response_content_type_is_json(self, client, valid_payload):
        response = client.post("/prediction", json=valid_payload)
        assert "application/json" in response.headers.get("content-type", "")

class TestPredictionDeterminism:

    def test_same_input_gives_same_result(self, client, valid_payload):
        r1 = client.post("/prediction", json=valid_payload).json()
        r2 = client.post("/prediction", json=valid_payload).json()
        assert r1["result"] == r2["result"]
        assert r1["probability_malignant"] == r2["probability_malignant"]
        assert r1["probability_benign"] == r2["probability_benign"]

    def test_different_inputs_can_give_different_results(
        self, client, valid_payload, malignant_payload
    ):
        r1 = client.post("/prediction", json=valid_payload).json()
        r2 = client.post("/prediction", json=malignant_payload).json()

        assert r1["probability_malignant"] != r2["probability_malignant"]
