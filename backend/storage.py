"""Helpers for fetching task assets."""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class StorageError(RuntimeError):
    """Raised when fetching remote assets fails."""


async def fetch_audio_bytes(url: str, *, timeout: Optional[httpx.Timeout] = None) -> bytes:
    """Download audio content from a URL and return raw bytes."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise StorageError(f"Only HTTP(S) URLs are supported; received: {url}")

    client_timeout = timeout or DEFAULT_TIMEOUT
    buffer = BytesIO()
    async with httpx.AsyncClient(timeout=client_timeout, follow_redirects=True) as client:
        try:
            async with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    body_preview = await response.aread()
                    raise StorageError(
                        f"Failed to retrieve {url}: HTTP {response.status_code} "
                        f"{body_preview[:200]!r}"
                    )

                async for chunk in response.aiter_bytes():
                    buffer.write(chunk)

        except httpx.HTTPError as exc:
            raise StorageError(f"Failed to retrieve {url}: {exc}") from exc

    return buffer.getvalue()


