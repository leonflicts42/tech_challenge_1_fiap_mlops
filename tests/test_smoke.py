"""Smoke tests — verificam que o serviço sobe e responde.

Não dependem de artefatos em disco (usam os mocks do conftest).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_app_starts(test_client: TestClient) -> None:
    assert test_client is not None


def test_health_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_schema(test_client: TestClient) -> None:
    data = test_client.get("/api/v1/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data
    assert "model_version" in data
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["preprocessor_loaded"], bool)


def test_health_with_mocks_is_healthy(test_client: TestClient) -> None:
    data = test_client.get("/api/v1/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_predict_endpoint_exists(test_client: TestClient, valid_payload: dict) -> None:
    response = test_client.post("/api/v1/predict", json=valid_payload)
    assert response.status_code == 200


def test_request_id_header_present(
    test_client: TestClient, valid_payload: dict
) -> None:
    response = test_client.post("/api/v1/predict", json=valid_payload)
    assert "x-request-id" in response.headers


def test_latency_header_present(test_client: TestClient, valid_payload: dict) -> None:
    response = test_client.post("/api/v1/predict", json=valid_payload)
    assert "x-process-time" in response.headers
