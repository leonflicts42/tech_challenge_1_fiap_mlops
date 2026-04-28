"""Testes do app FastAPI — lifespan, startup e rota raiz."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_check_route(test_client: TestClient) -> None:
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200


def test_root_retorna_200(test_client: TestClient) -> None:
    r = test_client.get("/")
    assert r.status_code == 200


def test_root_tem_link_docs(test_client: TestClient) -> None:
    body = test_client.get("/").json()
    assert "docs" in body


def test_startup_falha_graciosamente_sem_artefatos() -> None:
    """Quando os artefatos não existem, a API sobe com predictor=None (503 nos endpoints)."""
    import os
    from main import app

    os.environ["MODEL_PATH"] = "/nao/existe/modelo.pt"

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.predictor = None
        r = client.get("/api/v1/health")
        assert r.status_code == 503

    del os.environ["MODEL_PATH"]
