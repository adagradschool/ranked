from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modules.api import RagStore, app


def _pick_business_name(store: RagStore) -> str | None:
    data = store.collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    for metadata in metadatas:
        name = (metadata or {}).get("business_name")
        if name:
            return name
    return None


def test_gap_blog_flow_real_db(tmp_path):
    rag_dir = Path("rag_index")
    if not rag_dir.exists():
        pytest.skip("rag_index missing; run the scraper to build the DB.")

    store = RagStore.from_env()
    business_name = _pick_business_name(store)
    if not business_name:
        pytest.skip("No business names found in the RAG index.")

    app.state.store = store
    client = TestClient(app)

    content_response = client.post(
        "/business-content",
        json={
            "business_name": business_name,
            "max_documents": 50,
            "max_context_chars": 4000,
        },
    )
    assert content_response.status_code == 200
    content_payload = content_response.json()
    assert content_payload["matched_business_name"]
    assert content_payload["documents"]
    assert content_payload["context"]

    output_path = None
    try:
        blog_response = client.post(
            "/generate-blog",
            json={
                "business_name": business_name,
                "question": "What should a first-time visitor know?",
                "output_dir": str(tmp_path),
                "auto_index": False,
            },
        )
        assert blog_response.status_code == 200
        blog_payload = blog_response.json()
        assert blog_payload["indexed"] is False
        assert blog_payload["chunks_indexed"] == 0
        assert "<html" in blog_payload["html"].lower()

        blog_url = blog_payload["blog_url"]
        slug = blog_url.split("/")[-1]
        output_path = tmp_path / f"{slug}.html"
        assert output_path.exists()
    finally:
        if output_path and output_path.exists():
            output_path.unlink()
