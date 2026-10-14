"""
Inputs that reach the server's network or filesystem.

- /api/job/extract/url must not fetch internal addresses (SSRF), including
  via redirects.
- /api/cv/extract, which opened any file path on the server, is gone.

DNS is stubbed, so nothing here touches the network.

Run with::

    cd backend
    pytest tests/test_input_hardening.py -v
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import url_safety
from app.services.url_safety import UnsafeURLError, check_public_url, fetch_public_page


def _resolves_to(*addresses: str):
    infos = [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]
    return patch.object(url_safety.socket, "getaddrinfo", return_value=infos)


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.5", "172.16.3.4", "192.168.1.10", "169.254.169.254",
    "0.0.0.0", "::1", "fd00::1", "::ffff:127.0.0.1", "100.64.0.1",
])
def test_internal_addresses_are_refused(address):
    with _resolves_to(address), pytest.raises(UnsafeURLError):
        check_public_url("https://jobs.example.com/posting/1")


def test_a_host_with_any_internal_address_is_refused():
    with _resolves_to("93.184.215.14", "10.0.0.5"), pytest.raises(UnsafeURLError):
        check_public_url("https://jobs.example.com/")


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "https://user:pw@example.com/", "http:///nohost",
])
def test_non_web_or_credentialed_urls_are_refused(url):
    with _resolves_to("93.184.215.14"), pytest.raises(UnsafeURLError):
        check_public_url(url)


def test_public_address_is_allowed():
    with _resolves_to("93.184.215.14"):
        check_public_url("https://jobs.example.com/posting/1")


def _response(status: int, headers: dict, body: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.is_redirect = status in (301, 302, 303, 307, 308)
    resp.headers = headers
    resp.encoding = "utf-8"
    resp.iter_content.return_value = [body]
    resp.raise_for_status.return_value = None
    return resp


def test_redirect_to_an_internal_address_is_refused():
    def resolve(host, *_args, **_kwargs):
        address = "93.184.215.14" if host == "jobs.example.com" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    redirect = _response(302, {"Location": "http://localhost:5432/"})
    with patch.object(url_safety.socket, "getaddrinfo", side_effect=resolve), \
         patch.object(url_safety.requests, "get", return_value=redirect) as get:
        with pytest.raises(UnsafeURLError):
            fetch_public_page("https://jobs.example.com/posting/1")
    assert get.call_count == 1  # the internal target was never requested


def test_public_page_text_is_extracted():
    page = _response(200, {"Content-Type": "text/html"},
                     b"<html><script>x()</script><h1>Werkstudent Java</h1><p>Spring Boot</p></html>")
    with _resolves_to("93.184.215.14"), patch.object(url_safety.requests, "get", return_value=page):
        text = fetch_public_page("https://jobs.example.com/posting/1")
    assert "Werkstudent Java" in text and "Spring Boot" in text
    assert "x()" not in text


def test_job_url_endpoint_answers_400_for_internal_targets():
    with TestClient(app) as client, _resolves_to("169.254.169.254"):
        resp = client.post("/api/job/extract/url", json={"urls": ["http://metadata.internal/latest/"]})
    assert resp.status_code == 400
    assert "not a public address" in resp.json()["detail"]


def test_server_path_extraction_endpoint_is_gone():
    with TestClient(app) as client:
        resp = client.post("/api/cv/extract", json={"pdf_path": "C:/Windows/win.ini"})
    assert resp.status_code in (404, 405)
