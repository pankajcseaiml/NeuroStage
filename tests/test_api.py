"""Integration tests for FastAPI backend routes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    # Use TestClient without running long startup model download
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "NeuroStage API" in data["message"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data


def test_protected_route_without_token(client):
    # Attempting to access protected endpoints without auth returns 401
    response = client.get("/experiments")
    assert response.status_code in (401, 403)
