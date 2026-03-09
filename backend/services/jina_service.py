"""
Jina Reader Service

Wrapper for the Jina Reader API used by Agent 1 (Claim Extraction)
to scrape article content as clean markdown from a URL.
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)


class JinaService:
    """Jina Reader API wrapper for URL-to-markdown scraping."""

    def __init__(self):
        self.api_key = os.getenv("JINA_READER_API_KEY")
        self.base_url = "https://r.jina.ai"

    async def scrape_url(self, url: str) -> str:
        """
        Scrape a URL and return cleaned markdown content via Jina Reader.

        Args:
            url: The article URL to scrape.

        Returns:
            Markdown-formatted article content as a string.
        """
        headers = {
            "Accept": "text/markdown",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/{url}",
                headers=headers,
            )
            response.raise_for_status()
            return response.text
