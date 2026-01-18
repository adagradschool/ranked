import argparse
import hashlib
import json
import os
import logging
import re
from urllib.parse import urlparse

import chromadb
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from modules.resolver import (
    extract_candidate_names,
    extract_title,
    fetch_html,
    is_aggregator_url,
    resolve_entities_with_openai,
)
from modules.scraper import CrawlConfig, crawl_sites, extract_text

EXA_SEARCH_URL = "https://api.exa.ai/search"


def exa_search(api_key: str, query: str, num_results: int) -> list[dict]:
    payload = {
        "query": query,
        "num_results": num_results,
        "use_autoprompt": True,
    }
    headers = {"x-api-key": api_key, "content-type": "application/json"}
    response = requests.post(EXA_SEARCH_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    logging.info("EXA search: query=%r results=%d", query, len(results))
    return results


def chunk_text(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()


def create_rag_index(
    pages: list[dict],
    output_dir: str,
    collection_name: str,
    embedding_model: str,
    max_words: int,
    overlap: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=output_dir)
    collection = client.get_or_create_collection(name=collection_name)
    model = SentenceTransformer(embedding_model)

    ids = []
    documents = []
    metadatas = []

    for page in pages:
        chunks = chunk_text(page["text"], max_words=max_words, overlap=overlap)
        logging.debug("Chunking %s -> %d chunks", page.get("url"), len(chunks))
        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{page['url']}#{index}".encode("utf-8")).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "url": page["url"],
                    "domain": page.get("domain", ""),
                    "category": page.get("category", ""),
                    "business_url": page.get("business_url", ""),
                    "business_name": page.get("business_name", ""),
                    "queries": "; ".join(page.get("queries", [])),
                    "chunk_index": index,
                }
            )

    if not documents:
        return

    embeddings = embed_texts(model, documents)
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    logging.info("Indexed chunks: %d", len(documents))


def build_queries() -> dict[str, list[str]]:
    return {
        "healthcare": [
            "healthcare businesses in the United States",
            "primary care clinics and healthcare providers",
            "hospitals and medical centers",
            "medical practices and urgent care clinics",
        ],
        "nyc_restaurants": [
            "restaurants in New York City",
            "NYC dining and restaurant listings",
            "New York City restaurants",
            "best restaurants in NYC",
        ],
    }


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def pick_business_url(results: list[dict], entity: str, category: str) -> str:
    entity_token = normalize_token(entity)
    for result in results:
        candidate_url = result.get("url")
        if not candidate_url:
            continue
        if is_aggregator_url(candidate_url):
            continue
        domain = urlparse(candidate_url).netloc
        if category == "nyc_restaurants" and domain.endswith(".gov"):
            continue
        title = result.get("title", "") or ""
        if entity_token:
            if entity_token in normalize_token(title) or entity_token in normalize_token(candidate_url):
                return candidate_url
        else:
            return candidate_url
    for result in results:
        candidate_url = result.get("url")
        if not candidate_url:
            continue
        if is_aggregator_url(candidate_url):
            continue
        domain = urlparse(candidate_url).netloc
        if category == "nyc_restaurants" and domain.endswith(".gov"):
            continue
        return candidate_url
    return ""


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build a RAG index from EXA search results.")
    parser.add_argument("--output-dir", default="rag_index", help="Directory for Chroma DB")
    parser.add_argument("--collection", default="exa_rag", help="Chroma collection name")
    parser.add_argument("--num-results", type=int, default=25, help="EXA results per query")
    parser.add_argument("--target-healthcare", type=int, default=100)
    parser.add_argument("--target-nyc-restaurants", type=int, default=100)
    parser.add_argument("--max-pages-per-domain", type=int, default=5)
    parser.add_argument("--max-total-pages", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--output-json", default="rag_pages.json")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--max-entities-per-directory", type=int, default=25)
    parser.add_argument("--entity-search-results", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise SystemExit("EXA_API_KEY is required")

    queries = build_queries()
    category_targets = {
        "healthcare": args.target_healthcare,
        "nyc_restaurants": args.target_nyc_restaurants,
    }
    active_categories = {category for category, target in category_targets.items() if target > 0}
    if not active_categories:
        raise SystemExit("No categories selected; set target counts above zero.")
    logging.info("Active categories: %s", ", ".join(sorted(active_categories)))

    category_urls: dict[str, list[str]] = {category: [] for category in active_categories}
    url_queries: dict[str, set[str]] = {}
    for category, query_list in queries.items():
        if category not in active_categories:
            continue
        for query in query_list:
            results = exa_search(api_key, query, num_results=args.num_results)
            for item in results:
                url = item.get("url")
                if url:
                    category_urls[category].append(url)
                    url_queries.setdefault(url, set()).add(query)

    category_urls = {
        category: list(dict.fromkeys(urls)) for category, urls in category_urls.items()
    }

    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    if openai_client:
        logging.info("OpenAI entity resolution enabled")
    else:
        logging.info("OpenAI entity resolution disabled")

    business_candidates: list[dict[str, object]] = []
    directory_keywords = {
        "best",
        "top",
        "guide",
        "list",
        "directory",
        "restaurants",
        "companies",
        "category",
        "compare",
    }
    search_suffix = {
        "nyc_restaurants": "restaurant New York City official website",
        "healthcare": "clinic healthcare official website",
    }

    for category, urls in category_urls.items():
        for url in urls:
            html = fetch_html(url, timeout=args.timeout)
            if not html:
                logging.debug("Fetch failed: %s", url)
                continue
            title = extract_title(html)
            path = urlparse(url).path.lower()
            looks_like_directory = (
                is_aggregator_url(url)
                or any(keyword in title.lower() for keyword in directory_keywords)
                or any(keyword in path for keyword in directory_keywords)
            )

            if openai_client and looks_like_directory:
                logging.info("Directory detected: %s (title=%r, path=%r)", url, title, path)
                page_text = extract_text(html)
                candidate_names = extract_candidate_names(html, limit=200)
                logging.info(
                    "Directory text length=%d candidate_names=%d",
                    len(page_text),
                    len(candidate_names),
                )
                entities = resolve_entities_with_openai(
                    openai_client,
                    page_text,
                    candidate_names,
                    max_items=args.max_entities_per_directory,
                    model=args.openai_model,
                )
                logging.info("Resolved %d entities from %s", len(entities), url)
                if entities:
                    for entity in entities:
                        results = exa_search(
                            api_key,
                            f"{entity} {search_suffix.get(category, 'official website')}",
                            num_results=args.entity_search_results,
                        )
                        business_url = pick_business_url(results, entity, category)
                        if business_url:
                            logging.debug("Entity resolved: %s -> %s", entity, business_url)
                            business_candidates.append(
                                {
                                    "business_url": business_url,
                                    "business_name": entity,
                                    "category": category,
                                    "queries": set(url_queries.get(url, set())),
                                }
                            )
                    continue

            if looks_like_directory:
                logging.debug("Skipping directory without resolution: %s", url)
                continue

            business_name = title or urlparse(url).netloc
            logging.debug("Using direct business url: %s", url)
            business_candidates.append(
                {
                    "business_url": url,
                    "business_name": business_name,
                    "category": category,
                    "queries": set(url_queries.get(url, set())),
                }
            )

    selected_urls: list[str] = []
    domain_metadata: dict[str, dict[str, object]] = {}
    category_counts: dict[str, int] = {category: 0 for category in active_categories}
    for candidate in business_candidates:
        category = candidate["category"]
        if category_counts[category] >= category_targets[category]:
            continue
        business_url = candidate["business_url"]
        domain = urlparse(business_url).netloc
        if domain in domain_metadata:
            metadata = domain_metadata[domain]
            metadata.setdefault("queries", set()).update(candidate.get("queries", set()))
            continue
        domain_metadata[domain] = {
            "category": category,
            "business_url": business_url,
            "business_name": candidate.get("business_name", ""),
            "queries": set(candidate.get("queries", set())),
        }
        selected_urls.append(business_url)
        category_counts[category] += 1

    urls = list(dict.fromkeys(selected_urls))

    for metadata in domain_metadata.values():
        if isinstance(metadata.get("queries"), set):
            metadata["queries"] = sorted(metadata["queries"])
    logging.info("Selected domains: %d", len(domain_metadata))

    crawl_config = CrawlConfig(
        max_pages_per_domain=args.max_pages_per_domain,
        max_total_pages=args.max_total_pages,
        request_timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    pages = crawl_sites(urls, crawl_config, domain_metadata)
    logging.info("Crawled pages: %d", len(pages))

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(pages, handle, indent=2)

    create_rag_index(
        pages,
        output_dir=args.output_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        max_words=args.chunk_words,
        overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
