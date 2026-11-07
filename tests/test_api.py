from fastapi.testclient import TestClient
from api.main import app

def test_health_endpoint_is_defined():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code in (200, 503)
