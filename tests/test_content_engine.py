import os
import pandas as pd
import redis
import pytest
from content_engine import ContentEngine

FIXTURE_CSV = "tests/fixture.csv"

@pytest.fixture(scope="module")
def fixture_data(tmp_path):
    df = pd.DataFrame({
        "id": [1,2],
        "description": ["foo bar baz", "bar baz qux"]
    })
    path = tmp_path / FIXTURE_CSV
    df.to_csv(path, index=False)
    yield str(path)

@pytest.fixture(scope="module")
def engine(fixture_data):
    # use a fresh Redis DB
    e = ContentEngine(redis_url="redis://localhost:6379/1")
    e._r.flushdb()
    e.train(fixture_data)
    return e

def test_predict_returns_list(engine):
    recs = engine.predict(1, n=1)
    assert isinstance(recs, list)
    assert recs and isinstance(recs[0]["id"], int)

def test_predict_unknown_id(engine):
    assert engine.predict(999) == []
