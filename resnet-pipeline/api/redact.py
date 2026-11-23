"""Strip credentials from URLs before they reach logs or HTTP error bodies."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def redact_rtsp(url: str) -> str:
    """Return ``url`` with ``user:pass@`` removed when present."""
    text = str(url or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.username and not parts.password:
        return text
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
