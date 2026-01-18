set dotenv-load := true

target_healthcare := "100"
target_nyc_restaurants := "100"
num_results := "25"
max_pages_per_domain := "5"
max_total_pages := "200"
output_dir := "rag_index"
collection := "exa_rag"
embedding_model := "all-MiniLM-L6-v2"
chunk_words := "300"
chunk_overlap := "40"
sleep_seconds := "0"
timeout := "15"
openai_model := "gpt-4o-mini"
max_entities_per_directory := "25"
entity_search_results := "3"

scrape *ARGS:
    .venv/bin/python main.py \
        --target-healthcare {{target_healthcare}} \
        --target-nyc-restaurants {{target_nyc_restaurants}} \
        --num-results {{num_results}} \
        --max-pages-per-domain {{max_pages_per_domain}} \
        --max-total-pages {{max_total_pages}} \
        --output-dir {{output_dir}} \
        --collection {{collection}} \
        --embedding-model {{embedding_model}} \
        --chunk-words {{chunk_words}} \
        --chunk-overlap {{chunk_overlap}} \
        --sleep-seconds {{sleep_seconds}} \
        --timeout {{timeout}} \
        --openai-model {{openai_model}} \
        --max-entities-per-directory {{max_entities_per_directory}} \
        --entity-search-results {{entity_search_results}} \
        {{ARGS}}

inspect-chroma:
    .venv/bin/python -c 'import chromadb; client=chromadb.PersistentClient(path="{{output_dir}}"); col=client.get_collection("{{collection}}"); print("count", col.count()); print(col.get(limit=3, include=["metadatas", "documents"]))'

query-chroma QUERY:
    .venv/bin/python -c 'import sys, chromadb; client=chromadb.PersistentClient(path="{{output_dir}}"); col=client.get_collection("{{collection}}"); q=sys.argv[1]; print(col.query(query_texts=[q], n_results=3, include=["documents", "metadatas", "distances"]))' "{{QUERY}}"
