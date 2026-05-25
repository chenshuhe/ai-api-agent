"""Async OpenAPI doc fetcher with concurrent loading."""

import asyncio

import httpx
from loguru import logger

from .parser import ApiDoc, parse_openapi


async def fetch_doc(url: str, timeout: int = 30) -> dict:
    """Fetch an OpenAPI JSON doc from a URL."""
    async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def load_all(urls: list[str], timeout: int = 30) -> list[ApiDoc]:
    """Fetch and parse multiple OpenAPI docs concurrently."""
    results = list(await asyncio.gather(
        *[fetch_doc(url, timeout) for url in urls],
        return_exceptions=True,
    ))

    docs = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to fetch {url}: {result}")
            continue
        try:
            doc = parse_openapi(result)
            docs.append(doc)
            logger.info(f"Loaded {len(doc.endpoints)} endpoints from {url} ({doc.title})")
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
    return docs
