import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from modules.question_agent import generate_questions

load_dotenv()

RAG_PAGES_PATH = os.getenv("RAG_PAGES_PATH", "rag_pages.json")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

pages_data: list[dict] = []
openai_client: OpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pages_data, openai_client

    if os.path.exists(RAG_PAGES_PATH):
        with open(RAG_PAGES_PATH, encoding="utf-8") as f:
            pages_data = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)

    yield


app = FastAPI(
    title="Question Generation API",
    description="Generate questions from crawled RAG data using OpenAI",
    lifespan=lifespan,
)


class GenerateQuestionsRequest(BaseModel):
    page_url: str | None = None
    limit: int = 5
    num_queries: int = 10


class SearchQuery(BaseModel):
    query: str
    intent: str | None = None


class PageQueries(BaseModel):
    source_url: str
    business_name: str
    category: str
    queries: list[SearchQuery]


class GenerateQuestionsResponse(BaseModel):
    results: list[PageQueries]


@app.get("/pages")
def list_pages():
    return {
        "count": len(pages_data),
        "pages": [
            {
                "url": p.get("url"),
                "domain": p.get("domain"),
                "business_name": p.get("business_name"),
                "category": p.get("category"),
            }
            for p in pages_data
        ],
    }


@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
def generate_questions_endpoint(request: GenerateQuestionsRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")

    if not pages_data:
        raise HTTPException(status_code=404, detail="No pages data loaded")

    if request.page_url:
        target_pages = [p for p in pages_data if p.get("url") == request.page_url]
        if not target_pages:
            raise HTTPException(status_code=404, detail="Page not found")
    else:
        target_pages = pages_data[: request.limit]

    results = []
    for page in target_pages:
        text = page.get("text", "")
        business_name = page.get("business_name", "")
        category = page.get("category", "")
        queries = generate_questions(
            openai_client,
            text,
            business_name=business_name,
            category=category,
            model=OPENAI_MODEL,
            num_queries=request.num_queries,
        )
        results.append(
            PageQueries(
                source_url=page.get("url", ""),
                business_name=business_name,
                category=category,
                queries=[SearchQuery(**q) for q in queries],
            )
        )

    return GenerateQuestionsResponse(results=results)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "pages_loaded": len(pages_data),
        "openai_configured": openai_client is not None,
    }
