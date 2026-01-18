import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import chromadb

from modules.resolver import is_aggregator_url


BLOCKED_DOMAINS = {
    "alexreichek.com",
    "blog.resy.com",
    "guide.michelin.com",
    "metropolitanshuttle.com",
    "nymag.com",
    "ny.eater.com",
    "reddit.com",
    "theinfatuation.com",
}


BAD_NAME_PHRASES = {
    "privacy policy",
    "cookie policy",
    "terms of use",
    "where to eat",
    "hit list",
    "restaurant reviews",
    "dining guide",
    "food guide",
}


BAD_NAME_PAIRS = (
    ("best", "restaurant"),
    ("best", "restaurants"),
    ("top", "restaurant"),
    ("top", "restaurants"),
    ("guide", "restaurant"),
    ("guide", "restaurants"),
    ("list", "restaurant"),
    ("list", "restaurants"),
    ("directory", "restaurant"),
    ("directory", "restaurants"),
    ("review", "restaurant"),
    ("reviews", "restaurant"),
    ("reviews", "restaurants"),
    ("restaurants", "in"),
    ("restaurants", "of"),
    ("favorite", "restaurants"),
)


BAD_URL_TOKENS = {
    "privacy",
    "terms",
    "policy",
    "cookies",
}


@dataclass
class BusinessGroup:
    business_url: str
    ids: list[str]
    names: Counter


def normalize_domain(domain: str) -> str:
    domain = domain.lower()
    if domain.startswith("www."):
        return domain[4:]
    return domain


def domain_matches(domain: str, candidates: Iterable[str]) -> bool:
    domain = normalize_domain(domain)
    for candidate in candidates:
        candidate = normalize_domain(candidate)
        if domain == candidate or domain.endswith(f".{candidate}"):
            return True
    return False


def looks_like_directory_name(name: str) -> bool:
    lowered = name.lower().strip()
    if not lowered:
        return True
    if any(phrase in lowered for phrase in BAD_NAME_PHRASES):
        return True
    for left, right in BAD_NAME_PAIRS:
        if left in lowered and right in lowered:
            return True
    return False


def looks_like_policy_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(token in path for token in BAD_URL_TOKENS)


def group_by_business(collection: chromadb.Collection) -> dict[str, BusinessGroup]:
    groups: dict[str, BusinessGroup] = {}
    offset = 0
    batch = 1000
    while True:
        result = collection.get(limit=batch, offset=offset, include=["metadatas"])
        metadatas = result.get("metadatas", [])
        ids = result.get("ids", [])
        if not metadatas:
            break
        for doc_id, meta in zip(ids, metadatas):
            meta = meta or {}
            business_url = meta.get("business_url", "")
            business_name = meta.get("business_name", "")
            group = groups.get(business_url)
            if not group:
                group = BusinessGroup(business_url=business_url, ids=[], names=Counter())
                groups[business_url] = group
            group.ids.append(doc_id)
            if business_name:
                group.names[business_name] += 1
        offset += len(metadatas)
        if len(metadatas) < batch:
            break
    return groups


def is_blocked_business(business_url: str, name: str) -> tuple[bool, str]:
    if not business_url:
        return True, "missing business_url"
    if not name:
        return True, "missing business_name"
    if not business_url.startswith("http"):
        return True, "invalid business_url"
    if is_aggregator_url(business_url):
        return True, "aggregator domain"
    parsed = urlparse(business_url)
    domain = parsed.netloc
    if domain_matches(domain, BLOCKED_DOMAINS):
        return True, "blocked domain"
    if looks_like_directory_name(name):
        return True, "directory name"
    if looks_like_policy_url(business_url):
        return True, "policy url"
    return False, ""


def cleanup_chroma(output_dir: str, collection_name: str, apply: bool) -> None:
    client = chromadb.PersistentClient(path=output_dir)
    collection = client.get_collection(name=collection_name)
    groups = group_by_business(collection)

    deletions: list[str] = []
    reasons: Counter = Counter()
    for business_url, group in groups.items():
        name = group.names.most_common(1)[0][0] if group.names else ""
        blocked, reason = is_blocked_business(business_url, name)
        if blocked:
            deletions.extend(group.ids)
            reasons[reason] += len(group.ids)

    logging.info("Identified %d chunks to delete", len(deletions))
    for reason, count in reasons.most_common():
        logging.info("  %s: %d", reason, count)

    if deletions and apply:
        collection.delete(ids=deletions)
        logging.info("Deleted %d chunks", len(deletions))
    elif deletions:
        logging.info("Dry run: no deletions applied")
    else:
        logging.info("No deletions required")


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Clean a Chroma collection of non-business content.")
    parser.add_argument("--output-dir", default="rag_index", help="Directory for Chroma DB")
    parser.add_argument("--collection", default="exa_rag", help="Chroma collection name")
    parser.add_argument("--apply", action="store_true", help="Apply deletions")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cleanup_chroma(args.output_dir, args.collection, apply=args.apply)


if __name__ == "__main__":
    cli_main()
