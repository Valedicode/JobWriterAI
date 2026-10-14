"""
Fetching user-supplied job-posting URLs without exposing the server's network.

/api/job/extract/url used to hand any http(s) URL straight to WebBaseLoader,
so a request for http://localhost:5432/, http://169.254.169.254/ (cloud
metadata) or any internal host was fetched by the server and its contents fed
to the LLM: a server-side request forgery (SSRF) hole.

``fetch_public_page`` only connects to hosts whose every resolved address is
public, follows redirects one hop at a time (re-checking each target), and
caps time and size.

Remaining gap: the name is resolved for the check and again by the HTTP
client, so a DNS-rebinding attacker could in principle swap the address in
between. Closing that needs a connection pinned to the checked IP; if this
endpoint is ever exposed to untrusted users at scale, add that or route
fetches through an egress proxy.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

MAX_REDIRECTS = 5
MAX_BYTES = 5 * 1024 * 1024
TIMEOUT_S = 15
DEFAULT_USER_AGENT = "JobWriterAI/0.1 (job posting fetcher)"


class UnsafeURLError(ValueError):
    """The URL is malformed or points at a non-public address."""


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_public_url(url: str) -> None:
    """Raise UnsafeURLError unless ``url`` is http(s) to a host that resolves only to public addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs are allowed.")
    if not parts.hostname:
        raise UnsafeURLError("The URL has no host.")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with credentials are not allowed.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve {parts.hostname}.") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(_is_public(a) for a in addresses):
        raise UnsafeURLError(f"{parts.hostname} is not a public address.")


def fetch_public_page(url: str, user_agent: str | None = None) -> str:
    """
    The visible text of a public web page. Raises UnsafeURLError for
    non-public targets (including via redirects), requests exceptions for
    network/HTTP failures, and ValueError for oversized or non-text bodies.
    """
    headers = {"User-Agent": user_agent or DEFAULT_USER_AGENT}
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_public_url(current)
        response = requests.get(current, headers=headers, timeout=TIMEOUT_S, allow_redirects=False, stream=True)
        try:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect without a Location header.")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if content_type and "html" not in content_type and "text" not in content_type:
                raise ValueError(f"Not a web page ({content_type}).")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise ValueError("The page is too large.")
            text = body.decode(response.encoding or "utf-8", errors="replace")
        finally:
            response.close()
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    raise ValueError("Too many redirects.")
