"""Teste de smoke básico — verifica que o app importa e responde."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_check_route(test_client: TestClient) -> None:
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
