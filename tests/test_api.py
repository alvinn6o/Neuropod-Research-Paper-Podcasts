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


def test_logout_invalidates_session(client, auth_headers):
    assert client.get("/me", headers=auth_headers).status_code == 200
    assert client.post("/auth/logout", headers=auth_headers).status_code == 200
    assert client.get("/me", headers=auth_headers).status_code == 401


def test_categories_crud(client, auth_headers):
    response = client.get("/categories", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"categories": []}

    response = client.post(
        "/categories", headers=auth_headers,
        json={"categories": ["cs.AI", "cs.LG", "hep-th", "cs.AI"]},  # dupe
    )
    assert response.status_code == 200
    assert response.json()["categories"] == ["cs.AI", "cs.LG", "hep-th"]


def test_categories_reject_invalid(client, auth_headers):
    response = client.post(
        "/categories", headers=auth_headers,
        json={"categories": ["cs.AI", "with space", "with:colon", "valid-one"]},
    )
    # silently filters invalid entries
    cats = response.json()["categories"]
    assert "cs.AI" in cats
    assert "valid-one" in cats
    assert "with space" not in cats
    assert "with:colon" not in cats


def test_feed_returns_xml(client, auth_headers):
    slug = client.get("/me", headers=auth_headers).json()["feed_slug"]
    response = client.get(f"/feed/{slug}")
    assert response.status_code == 200
    assert "application/rss+xml" in response.headers["content-type"]
    assert b"<rss" in response.content


def test_feed_unknown_slug_404s(client):
    assert client.get("/feed/no-such-user").status_code == 404
