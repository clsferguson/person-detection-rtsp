import pytest
from app.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    rv = client.get('/')
    assert b'Person Detection' in rv.data

def test_config(client):
    rv = client.get('/config')
    assert b'Configuration' in rv.data
