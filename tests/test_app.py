import json
import pytest
from app import app, engine

@pytest.fixture
def client():
    engine._r.flushdb()
    # preload minimal data
    engine._r.zadd("p:smlr:1", {"2":0.9})
    with app.test_client() as c:
        yield c

def test_recommend_endpoint_success(client):
    resp = client.get("/recommend/1?n=1")
    data = json.loads(resp.data)
    assert resp.status_code == 200
    assert data == [{"id":2,"score":0.9}]

def test_recommend_endpoint_not_numeric(client):
    resp = client.get("/recommend/foo")
    assert resp.status_code == 404
