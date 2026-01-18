import hashlib
import html
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import chromadb
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


DEFAULT_RAG_DIR = "rag_index"
DEFAULT_COLLECTION = "exa_rag"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OUTPUT_DIR = "generated_blogs"


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "blog"
    return slug[:80]


def extract_visible_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks


def build_blog_prompt(business_name: str, question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write SEO-focused HTML blogs for business websites. "
                "Use only the provided context for facts. "
                "Return valid HTML5 with head, title, meta description, and a single article body. "
                "Include H1, multiple H2 sections, a short FAQ with the user question, and a clear call to action. "
                "Avoid markdown and do not wrap the HTML in code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Business name: {business_name}\n"
                "User question gap: {question}\n"
                "Context:\n{context}"
            ).format(business_name=business_name, question=question, context=context),
        },
    ]


def build_rag_chat_prompt(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are RankedGPT, a concise assistant for business research. "
                "Answer using only the provided context. "
                "Return the response in markdown. "
                "If the answer is not in the context, say you do not have enough data."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]


def extract_fallback_answer(context: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\\s+", context.strip())
    trimmed = [s for s in sentences if s]
    if not trimmed:
        return "I do not have enough data to answer that based on the current index."
    return " ".join(trimmed[:max_sentences])


def render_fallback_html(business_name: str, question: str, context: str) -> str:
    safe_name = html.escape(business_name)
    safe_question = html.escape(question)
    snippet = html.escape(context[:700]) if context else "Details forthcoming."
    return (
        "<!doctype html>"
        "<html lang=\"en\">"
        "<head>"
        f"<title>{safe_name} | Helpful Guide</title>"
        f"<meta name=\"description\" content=\"Learn about {safe_name} and answers to {safe_question}.\">"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "</head>"
        "<body>"
        f"<article><h1>{safe_name}: {safe_question}</h1>"
        "<p>This guide addresses a common customer question and highlights what to expect.</p>"
        "<h2>What to know</h2>"
        f"<p>{snippet}</p>"
        "<h2>Frequently asked questions</h2>"
        f"<p><strong>Q:</strong> {safe_question}</p>"
        "<p><strong>A:</strong> Contact the business for the most up-to-date details.</p>"
        "<h2>Next steps</h2>"
        f"<p>Reach out to {safe_name} to learn more or schedule a visit.</p>"
        "</article>"
        "</body>"
        "</html>"
    )


@dataclass
class RagStore:
    collection: Any
    embedding_model: SentenceTransformer
    openai_client: OpenAI | None
    openai_model: str
    business_names: list[str] | None = None

    @classmethod
    def from_env(cls) -> "RagStore":
        rag_dir = os.environ.get("RAG_INDEX_DIR", DEFAULT_RAG_DIR)
        collection_name = os.environ.get("RAG_COLLECTION", DEFAULT_COLLECTION)
        embedding_model = os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        openai_model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        client = chromadb.PersistentClient(path=rag_dir)
        collection = client.get_or_create_collection(name=collection_name)
        model = SentenceTransformer(embedding_model)
        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=openai_key) if openai_key else None
        return cls(
            collection=collection,
            embedding_model=model,
            openai_client=openai_client,
            openai_model=openai_model,
        )

    def refresh_business_names(self) -> None:
        data = self.collection.get(include=["metadatas"])
        names = []
        for metadata in data.get("metadatas", []) or []:
            name = (metadata or {}).get("business_name", "")
            if name:
                names.append(name)
        self.business_names = sorted(set(names))

    def resolve_business_name(self, business_name: str) -> str:
        if not self.business_names:
            self.refresh_business_names()
        if not self.business_names:
            return business_name
        target = normalize_name(business_name)
        best_name = ""
        best_score = 0.0
        for candidate in self.business_names:
            candidate_norm = normalize_name(candidate)
            if not candidate_norm:
                continue
            score = SequenceMatcher(None, target, candidate_norm).ratio()
            if target in candidate_norm or candidate_norm in target:
                score = max(score, 0.9)
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_score < 0.55:
            return business_name
        return best_name


class BusinessContentRequest(BaseModel):
    business_name: str = Field(..., min_length=1)
    max_documents: int = Field(200, ge=1, le=2000)
    max_context_chars: int = Field(12000, ge=500, le=50000)


class BusinessContentResponse(BaseModel):
    business_name: str
    matched_business_name: str
    documents: list[str]
    sources: list[str]
    context: str


class GenerateBlogRequest(BaseModel):
    business_name: str = Field(..., min_length=1)
    question: str = Field(..., min_length=5)
    slug: str | None = None
    blog_url: str | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    auto_index: bool = True
    max_documents: int = Field(200, ge=1, le=2000)
    max_context_chars: int = Field(12000, ge=500, le=50000)


class GenerateBlogResponse(BaseModel):
    business_name: str
    matched_business_name: str
    question: str
    blog_url: str
    html: str
    indexed: bool
    chunks_indexed: int


class ReindexBlogRequest(BaseModel):
    html: str = Field(..., min_length=20)
    blog_url: str = Field(..., min_length=5)
    business_name: str = Field(..., min_length=1)
    question: str | None = None


class ReindexBlogResponse(BaseModel):
    blog_url: str
    chunks_indexed: int


class QueryChromaRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=20)
    max_words: int = Field(100, ge=10, le=500)


class QueryChromaResult(BaseModel):
    business_url: str
    chunk: str
    distance: float | None = None
    score: float | None = None


class QueryChromaTopItem(BaseModel):
    business_url: str
    score: float


class QueryChromaResponse(BaseModel):
    query: str
    results: list[QueryChromaResult]
    top_list: list[QueryChromaTopItem]


class BusinessListResponse(BaseModel):
    businesses: list[str]


class RagChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(6, ge=1, le=20)
    max_context_chars: int = Field(6000, ge=500, le=20000)


class RagChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Ranked Blog API")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    load_dotenv()
    app.state.store = RagStore.from_env()
    app.state.store.refresh_business_names()
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@app.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


def get_store() -> RagStore:
    store = getattr(app.state, "store", None)
    if store is None:
        store = RagStore.from_env()
        app.state.store = store
    return store


def fetch_business_documents(
    store: RagStore,
    business_name: str,
    max_documents: int,
    max_context_chars: int,
) -> tuple[str, list[str], list[str], str]:
    matched_name = store.resolve_business_name(business_name)
    data = store.collection.get(
        where={"business_name": {"$eq": matched_name}},
        include=["documents", "metadatas"],
    )
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not documents:
        return matched_name, [], [], ""

    items = list(zip(documents, metadatas))
    items.sort(
        key=lambda item: (
            (item[1] or {}).get("url", ""),
            (item[1] or {}).get("chunk_index", 0),
        )
    )
    trimmed_docs = [doc for doc, _ in items][:max_documents]
    sources = []
    for _, metadata in items:
        url = (metadata or {}).get("url")
        if url and url not in sources:
            sources.append(url)

    context_parts = []
    size = 0
    seen = set()
    for doc in trimmed_docs:
        if doc in seen:
            continue
        seen.add(doc)
        if size + len(doc) > max_context_chars:
            remaining = max_context_chars - size
            if remaining > 0:
                context_parts.append(doc[:remaining])
            break
        context_parts.append(doc)
        size += len(doc)
    context = "\n".join(context_parts)
    return matched_name, trimmed_docs, sources, context


def index_blog_html(
    store: RagStore,
    html_text: str,
    blog_url: str,
    business_name: str,
    question: str | None,
    max_words: int = 260,
    overlap: int = 40,
) -> int:
    text = extract_visible_text(html_text)
    chunks = chunk_text(text, max_words=max_words, overlap=overlap)
    if not chunks:
        return 0
    ids = []
    documents = []
    metadatas = []
    for index, chunk in enumerate(chunks):
        chunk_id = hashlib.sha256(f"{blog_url}#{index}".encode("utf-8")).hexdigest()
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(
            {
                "url": blog_url,
                "domain": "",
                "category": "generated_blog",
                "business_url": "",
                "business_name": business_name,
                "queries": question or "",
                "chunk_index": index,
            }
        )
    embeddings = store.embedding_model.encode(documents, convert_to_numpy=True).tolist()
    if hasattr(store.collection, "upsert"):
        store.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    else:
        store.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    store.refresh_business_names()
    return len(documents)


@app.post("/business-content", response_model=BusinessContentResponse)
def business_content(payload: BusinessContentRequest) -> BusinessContentResponse:
    store = get_store()
    matched_name, documents, sources, context = fetch_business_documents(
        store,
        payload.business_name,
        payload.max_documents,
        payload.max_context_chars,
    )
    if not documents:
        raise HTTPException(status_code=404, detail="Business not found in RAG index.")
    return BusinessContentResponse(
        business_name=payload.business_name,
        matched_business_name=matched_name,
        documents=documents,
        sources=sources,
        context=context,
    )


@app.post("/generate-blog", response_model=GenerateBlogResponse)
def generate_blog(payload: GenerateBlogRequest) -> GenerateBlogResponse:
    store = get_store()
    matched_name, _, sources, context = fetch_business_documents(
        store,
        payload.business_name,
        payload.max_documents,
        payload.max_context_chars,
    )
    if not context:
        raise HTTPException(status_code=404, detail="Business not found in RAG index.")

    if store.openai_client:
        response = store.openai_client.responses.create(
            model=store.openai_model,
            input=build_blog_prompt(matched_name, payload.question, context),
        )
        html_text = response.output_text or ""
        if "```" in html_text:
            html_text = html_text.replace("```html", "").replace("```", "").strip()
        if not html_text:
            html_text = render_fallback_html(matched_name, payload.question, context)
    else:
        html_text = render_fallback_html(matched_name, payload.question, context)

    slug = slugify(payload.slug or f"{matched_name}-{payload.question}")
    blog_url = payload.blog_url or f"local://blog/{slug}"

    if sources:
        html_text = html_text + "\n<!-- Sources: " + ", ".join(sources) + " -->"

    output_dir = Path(payload.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug}.html"
    output_path.write_text(html_text, encoding="utf-8")

    indexed = False
    chunks_indexed = 0
    if payload.auto_index:
        chunks_indexed = index_blog_html(
            store,
            html_text=html_text,
            blog_url=blog_url,
            business_name=matched_name,
            question=payload.question,
        )
        indexed = chunks_indexed > 0

    return GenerateBlogResponse(
        business_name=payload.business_name,
        matched_business_name=matched_name,
        question=payload.question,
        blog_url=blog_url,
        html=html_text,
        indexed=indexed,
        chunks_indexed=chunks_indexed,
    )


@app.post("/reindex-blog", response_model=ReindexBlogResponse)
def reindex_blog(payload: ReindexBlogRequest) -> ReindexBlogResponse:
    store = get_store()
    chunks_indexed = index_blog_html(
        store,
        html_text=payload.html,
        blog_url=payload.blog_url,
        business_name=payload.business_name,
        question=payload.question,
    )
    if not chunks_indexed:
        raise HTTPException(status_code=400, detail="No indexable content in HTML.")
    return ReindexBlogResponse(blog_url=payload.blog_url, chunks_indexed=chunks_indexed)


@app.post("/query-chroma", response_model=QueryChromaResponse)
def query_chroma(payload: QueryChromaRequest) -> QueryChromaResponse:
    store = get_store()
    fetch_k = min(max(payload.top_k * 3, payload.top_k), 50)
    result = store.collection.query(
        query_texts=[payload.query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    results: list[QueryChromaResult] = []
    seen_businesses: set[str] = set()
    scored_items: list[tuple[str, float]] = []
    for doc, meta, distance in zip(docs, metas, distances):
        business_url = (meta or {}).get("business_url", "")
        business_key = business_url or "(none)"
        if business_key in seen_businesses:
            continue
        chunk = " ".join((doc or "").split()[: payload.max_words])
        score = float(distance) if distance is not None else None
        results.append(
            QueryChromaResult(
                business_url=business_key,
                chunk=chunk,
                distance=distance,
                score=None,
            )
        )
        if score is not None:
            scored_items.append((business_key, score))
        seen_businesses.add(business_key)
        if len(results) >= payload.top_k:
            break
    if scored_items:
        distances_only = [item[1] for item in scored_items]
        min_distance = min(distances_only)
        max_distance = max(distances_only)
        if max_distance == min_distance:
            normalized_scores = {item[0]: 1.0 for item in scored_items}
        else:
            normalized_scores = {
                item[0]: 1.0 - ((item[1] - min_distance) / (max_distance - min_distance))
                for item in scored_items
            }
        for item in results:
            if item.business_url in normalized_scores:
                item.score = round(normalized_scores[item.business_url], 6)
        top_list = [
            QueryChromaTopItem(business_url=url, score=score)
            for url, score in sorted(
                normalized_scores.items(), key=lambda pair: pair[1], reverse=True
            )
        ]
    else:
        top_list = []
    return QueryChromaResponse(query=payload.query, results=results, top_list=top_list)


@app.get("/businesses", response_model=BusinessListResponse)
def list_businesses(
    q: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=20, ge=1, le=200),
) -> BusinessListResponse:
    store = get_store()
    if not store.business_names:
        store.refresh_business_names()
    names = store.business_names or []
    if q:
        token = normalize_name(q)
        filtered = []
        for name in names:
            if token in normalize_name(name):
                filtered.append(name)
        names = filtered
    return BusinessListResponse(businesses=names[:limit])


@app.post("/rag-chat", response_model=RagChatResponse)
def rag_chat(payload: RagChatRequest) -> RagChatResponse:
    store = get_store()
    if not store.openai_client:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            store.openai_client = OpenAI(api_key=openai_key)
    result = store.collection.query(
        query_texts=[payload.question],
        n_results=max(payload.top_k * 3, payload.top_k),
        include=["documents", "metadatas"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    if not docs:
        raise HTTPException(status_code=404, detail="No matching context found.")

    sources = []
    context_parts = []
    size = 0
    for doc, meta in zip(docs, metas):
        if doc:
            context_parts.append(doc)
            size += len(doc)
        url = (meta or {}).get("url")
        if url and url not in sources:
            sources.append(url)
        if size >= payload.max_context_chars:
            break
    context = "\n".join(context_parts)[: payload.max_context_chars]

    answer = ""
    if store.openai_client:
        response = store.openai_client.responses.create(
            model=store.openai_model,
            input=build_rag_chat_prompt(payload.question, context),
        )
        answer = (response.output_text or "").strip()

    if not answer:
        answer = extract_fallback_answer(context)

    if sources:
        sources_md = "\n".join(f"- {source}" for source in sources)
        answer = f"{answer}\n\nSources:\n{sources_md}"
    return RagChatResponse(question=payload.question, answer=answer, sources=sources)
