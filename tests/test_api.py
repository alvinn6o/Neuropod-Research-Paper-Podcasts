"""API smoke + integration tests. No external API calls."""
from __future__ import annotations


def test_health_returns_ok(client):
    assert client.get("/health").status_code == 200


def test_healthz_verifies_db(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unauthenticated_protected_route_returns_401(client):
    assert client.get("/episodes").status_code == 401
    assert client.get("/me").status_code == 401
    assert client.post("/topics", json={"topics": ["llm"]}).status_code == 401


def test_request_id_header_round_trips(client):
    response = client.get("/health", headers={"X-Request-Id": "test-abc-123"})
    assert response.headers.get("X-Request-Id") == "test-abc-123"


def test_stub_login_creates_user_and_session(client):
    response = client.post("/auth/stub/login", json={"email": "newuser@example.com"})
    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["user"]["email"] == "newuser@example.com"
    assert body["user"]["feed_slug"] == "newuser"


def test_me_returns_logged_in_user(client, auth_headers):
    response = client.get("/me", headers=auth_headers)
    assert response.status_code == 200
    assert "@example.com" in response.json()["email"]


def test_topic_crud(client, auth_headers):
    response = client.post("/topics", headers=auth_headers, json={"topics": ["llm", "rag", "agents"]})
    assert response.status_code == 200
    assert response.json()["topics"] == ["llm", "rag", "agents"]

    response = client.get("/topics", headers=auth_headers)
    assert response.json()["topics"] == ["llm", "rag", "agents"]


def test_topic_dedupes_and_trims(client, auth_headers):
    response = client.post(
        "/topics", headers=auth_headers,
        json={"topics": ["LLM", "llm", "  rag  ", "rag", ""]},
    )
    topics = response.json()["topics"]
    assert len(topics) == 2
    assert "rag" in topics


def test_set_and_delete_provider_key(client, auth_headers):
    response = client.put(
        "/me/keys", headers=auth_headers,
        json={"provider": "openai", "api_key": "sk-test-1234567890"},
    )
    assert response.status_code == 200
    assert response.json()["keys"] == {"openai": "7890"}

    response = client.delete("/me/keys/openai", headers=auth_headers)
    assert response.json()["keys"] == {}


def test_bedrock_key_requires_full_credentials(client, auth_headers):
    response = client.put(
        "/me/keys", headers=auth_headers,
        json={"provider": "bedrock", "access_key": "AKIA", "secret_key": "12345678"},
    )
    assert response.status_code == 422


def test_bedrock_key_stores_region_in_hint(client, auth_headers):
    response = client.put(
        "/me/keys", headers=auth_headers,
        json={
            "provider": "bedrock",
            "region": "us-west-2",
            "access_key": "AKIAEXAMPLEEXAMPLE",
            "secret_key": "exampleSecretWith20PlusCharacters",
        },
    )
    assert response.status_code == 200
    hint = response.json()["keys"]["bedrock"]
    assert hint.startswith("us-west-2")
    assert "MPLE" in hint  # last 4 of access key masked


def test_unknown_provider_rejected(client, auth_headers):
    response = client.put("/me/keys", headers=auth_headers, json={"provider": "evil", "api_key": "sk-x" * 5})
    assert response.status_code == 422


def test_logout_invalidates_session(client, auth_headers):
    assert client.get("/me", headers=auth_headers).status_code == 200
    assert client.post("/auth/logout", headers=auth_headers).status_code == 200
    assert client.get("/me", headers=auth_headers).status_code == 401


def test_feed_returns_xml(client, auth_headers):
    slug = client.get("/me", headers=auth_headers).json()["feed_slug"]
    response = client.get(f"/feed/{slug}")
    assert response.status_code == 200
    assert "application/rss+xml" in response.headers["content-type"]
    assert b"<rss" in response.content


def test_feed_unknown_slug_404s(client):
    assert client.get("/feed/no-such-user").status_code == 404
