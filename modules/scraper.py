import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_UA = "ranked-rag/0.1 (+https://local)"
EXCLUDED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".zip",
    ".mp4",
    ".mp3",
}


@dataclass
class CrawlConfig:
    max_pages_per_domain: int
    max_total_pages: int
    request_timeout: float
    sleep_seconds: float


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return True
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in EXCLUDED_EXTENSIONS)


def extract_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        joined = urljoin(base_url, href)
        if should_skip_url(joined):
            continue
        if urlparse(joined).netloc != allowed_domain:
            continue
        links.append(joined)
    return links


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_space(text)


def crawl_domain(seed_urls: Iterable[str], config: CrawlConfig) -> list[dict]:
    seen = set()
    queue = list(seed_urls)
    pages = []
    allowed_domain = urlparse(queue[0]).netloc if queue else ""

    while queue and len(pages) < config.max_pages_per_domain:
        url = queue.pop(0)
        if url in seen or should_skip_url(url):
            continue
        seen.add(url)

        try:
            response = requests.get(
                url, headers={"user-agent": DEFAULT_UA}, timeout=config.request_timeout
            )
            if "text/html" not in response.headers.get("content-type", ""):
                continue
            html = response.text
        except requests.RequestException:
            continue

        text = extract_text(html)
        if text:
            pages.append({"url": url, "text": text})

        links = extract_links(html, url, allowed_domain)
        queue.extend(links)
        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)

    return pages


def crawl_sites(
    urls: Iterable[str],
    config: CrawlConfig,
    domain_metadata: dict[str, dict[str, object]],
) -> list[dict]:
    by_domain: dict[str, list[str]] = {}
    for url in urls:
        if should_skip_url(url):
            continue
        domain = urlparse(url).netloc
        by_domain.setdefault(domain, []).append(url)

    all_pages: list[dict] = []
    for domain, seeds in by_domain.items():
        if len(all_pages) >= config.max_total_pages:
            break
        remaining = config.max_total_pages - len(all_pages)
        per_domain_limit = min(config.max_pages_per_domain, remaining)
        pages = crawl_domain(
            seeds,
            CrawlConfig(
                max_pages_per_domain=per_domain_limit,
                max_total_pages=config.max_total_pages,
                request_timeout=config.request_timeout,
                sleep_seconds=config.sleep_seconds,
            ),
        )
        for page in pages:
            page["domain"] = domain
            metadata = domain_metadata.get(domain, {})
            page["category"] = metadata.get("category", "")
            page["business_url"] = metadata.get("business_url", "")
            page["business_name"] = metadata.get("business_name", "")
            page["queries"] = metadata.get("queries", [])
        all_pages.extend(pages)
    return all_pages
