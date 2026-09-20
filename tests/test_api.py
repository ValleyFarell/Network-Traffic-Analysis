"""Интеграционные проверки HTTP-маршрутов на лёгком модельном двойнике."""

from fastapi.testclient import TestClient

from nad_similarity.api import create_app
from tests.test_service import make_service


def new_host_payload(host_id: str = "X") -> dict:
    return {
        "host_id": host_id,
        "neighbors": 1,
        "flows": [
            {
                "time": 1,
                "duration": 2,
                "src": host_id,
                "src_port": "12345",
                "dst": "B",
                "dst_port": "80",
                "protocol": "6",
                "packets": 3,
                "bytes": 100,
            }
        ],
    }


def test_health_and_known_host_routes() -> None:
    with TestClient(create_app(service=make_service())) as client:
        health = client.get("/health")
        host = client.get("/hosts/A")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert host.status_code == 200
    assert host.json()["hierarchical_group"] == "G2.B0"


def test_predict_route() -> None:
    with TestClient(create_app(service=make_service())) as client:
        response = client.post("/hosts/predict", json=new_host_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["cluster_id"] == 4
    assert body["graph_role"] is None
    assert body["neighbors"][0]["host_id"] == "B"


def test_predict_rejects_known_host() -> None:
    with TestClient(create_app(service=make_service())) as client:
        response = client.post("/hosts/predict", json=new_host_payload("A"))

    assert response.status_code == 409


def test_predict_rejects_flow_unrelated_to_host() -> None:
    payload = new_host_payload()
    payload["flows"][0]["src"] = "A"

    with TestClient(create_app(service=make_service())) as client:
        response = client.post("/hosts/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_time_outside_training_period() -> None:
    payload = new_host_payload()
    payload["flows"][0]["time"] = 15 * 86_400 + 1

    with TestClient(create_app(service=make_service())) as client:
        response = client.post("/hosts/predict", json=payload)

    assert response.status_code == 422


def test_unknown_known_host_route_returns_404() -> None:
    with TestClient(create_app(service=make_service())) as client:
        response = client.get("/hosts/UNKNOWN")

    assert response.status_code == 404


def test_openapi_describes_all_public_routes() -> None:
    with TestClient(create_app(service=make_service())) as client:
        schema = client.get("/openapi.json").json()

    assert schema["info"]["description"]
    assert set(schema["paths"]) == {
        "/health",
        "/hosts/compare",
        "/hosts/predict",
        "/hosts/{host_id}",
        "/hosts/{host_id}/similar",
    }
    for operations in schema["paths"].values():
        for operation in operations.values():
            assert operation["summary"]
            assert operation["description"]
