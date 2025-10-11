import pytest
from app.main import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Real-Time Person Detection' in response.data


def test_config_get(client):
    response = client.get('/config')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'config' in payload


def test_config_post_validation(client):
    response = client.post('/config', json={'polygon': [[0, 0], [100, 0]]})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload['status'] == 'error'


def test_config_post_success(client):
    payload = {
        'rtsp_url': 'rtsp://example.com/stream',
        'polygon': [[0, 0], [100, 0], [100, 100], [0, 100]],
        'point': [50, 50],
        'max_dist': 200,
    }
    response = client.post('/config', json=payload)
    assert response.status_code == 200
    updated = response.get_json()
    assert updated['status'] == 'ok'
    assert updated['config']['rtsp_url'] == payload['rtsp_url']
