import json
import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from modules.scraper import extract_text, should_skip_url


AGGREGATOR_DOMAINS = {
    "www.yelp.com",
    "yelp.com",
    "www.tripadvisor.com",
    "tripadvisor.com",
    "www.opentable.com",
    "opentable.com",
    "www.zomato.com",
    "zomato.com",
    "www.doordash.com",
    "doordash.com",
    "www.grubhub.com",
    "grubhub.com",
    "www.seamless.com",
    "seamless.com",
    "www.ubereats.com",
    "ubereats.com",
    "www.eater.com",
    "eater.com",
    "www.timeout.com",
    "timeout.com",
    "www.healthgrades.com",
    "healthgrades.com",
    "www.zocdoc.com",
    "zocdoc.com",
    "www.webmd.com",
    "webmd.com",
    "www.vitals.com",
    "vitals.com",
    "www.psychologytoday.com",
    "psychologytoday.com",
    "www.md.com",
    "md.com",
    "www.doctor.com",
    "doctor.com",
}


def is_aggregator_url(url: str) -> bool:
    return urlparse(url).netloc in AGGREGATOR_DOMAINS


def fetch_html(url: str, timeout: float) -> str | None:
    try:
        response = requests.get(
            url, timeout=timeout, headers={"user-agent": "ranked-rag/0.1 (+https://local)"}
        )
        if "text/html" not in response.headers.get("content-type", ""):
            return None
        return response.text
    except requests.RequestException:
        return None


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    return (title or "").strip()


def extract_candidate_names(html: str, limit: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for tag in soup.find_all("a"):
        text = (tag.get_text() or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < 3 or len(text) > 80:
            continue
        if any(char.isdigit() for char in text):
            continue
        if any(word in text.lower() for word in ("best", "top", "guide", "restaurants")):
            continue
        candidates.append(text)
        if len(candidates) >= limit:
            break
    return list(dict.fromkeys(candidates))


def resolve_entities_with_openai(
    client: OpenAI,
    page_text: str,
    candidate_names: list[str],
    max_items: int,
    model: str,
) -> list[str]:
    prompt = {
        "page_text": page_text[:6000],
        "candidate_names": candidate_names[:200],
        "max_items": max_items,
    }
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Extract unique business or organization names from directory or list content. "
                    "Prefer names from candidate_names when present. Return only entity names (no URLs). "
                    "If unsure, omit. Limit to max_items."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given page_text and candidate_names, return JSON with an 'entities' array of strings."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
    )
    text = response.output_text or ""
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logging.info("OpenAI JSON parse failed: %s", text[:300])
        if candidate_names:
            logging.info("Falling back to candidate names (count=%d)", len(candidate_names))
            return candidate_names[:max_items]
        return []
    entities = data.get("entities", [])
    results = []
    for item in entities:
        name = (item or "").strip()
        if not name:
            continue
        results.append(name)
    if not results and candidate_names:
        logging.info("Falling back to candidate names (count=%d)", len(candidate_names))
        return candidate_names[:max_items]
    return results[:max_items]
