"""
Web scraping service.

Uses trafilatura for main content extraction + BeautifulSoup as fallback.
Trafilatura is state-of-the-art for web content extraction (precision >90% on benchmarks).
"""

import logging
import time
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from app.config import scraper_cfg

logger = logging.getLogger(__name__)


def scrape_url(url: str) -> Dict[str, Any]:
    """Scrape a URL and return structured content."""
    start = time.time()

    headers = {
        "User-Agent": scraper_cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,ja;q=0.7",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=scraper_cfg.timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise ValueError(f"Failed to fetch URL: {e}")

    html = response.text[:scraper_cfg.max_content_length]

    # Primary extraction via trafilatura (reuse already-fetched HTML, no double request)
    text = None
    title = None
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        metadata = trafilatura.bare_extraction(html, include_comments=False)
        if metadata and isinstance(metadata, dict):
            title = metadata.get("title")
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}")

    # Fallback: BeautifulSoup
    if not text:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""

        # Try article/main content first
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    # Also extract title from BeautifulSoup if not found
    if not title:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title else url

    # Clean up
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    elapsed = time.time() - start
    logger.info(f"Scraped {url} in {elapsed:.1f}s → {len(text)} chars")

    return {
        "url": url,
        "title": title or url,
        "text": text,
        "char_count": len(text),
        "elapsed_seconds": round(elapsed, 2),
    }
