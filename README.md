## ranked

Build a local RAG index from EXA search results. The pipeline targets healthcare businesses and NYC restaurants, crawls site content, chunks it, embeds it, and stores the vectors in a local Chroma DB.

### Requirements

- Python 3.12+
- `uv` package manager
- EXA API key

### Setup

Create a virtual environment and install dependencies:

```bash
uv venv
uv pip install -e .
```

Set your EXA API key in `.env`:

```bash
EXA_API_KEY=your_key_here
```

Optional: add `OPENAI_API_KEY` to enable entity resolution for directory/aggregator pages and follow-up searches for official business websites.

### Run

Using `just`:

```bash
just scrape --output-dir rag_index --collection exa_rag
```

Or directly:

```bash
.venv/bin/python main.py --output-dir rag_index --collection exa_rag
```

### Notes

- Crawling is limited by `--max-pages-per-domain` and `--max-total-pages`.
- Business targets can be set with `--target-healthcare` and `--target-nyc-restaurants`.
- Raw crawled pages are written to `rag_pages.json`.
- The raw crawled pages are written to `rag_pages.json` for inspection.
