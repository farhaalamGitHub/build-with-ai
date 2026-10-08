"""Hybrid retrieval: vector route + Atlas Search keyword route, fused by Reciprocal Rank Fusion.

Both routes read the same chunks/embeddings as semantic mode. Fusion uses ranks only, because
BM25 and cosine scores are not comparable. Failures are never silently degraded to semantic-only.
"""

from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import get_settings

RRF_K = 60
WEIGHTS = {"semantic": 1.0, "keyword": 1.0}
KEYWORD_PATH = "text"
_ready = False


def route_depth(limit: int) -> int:
    return max(limit, min(50, max(20, 4 * limit)))


def fuse(semantic_hits: list[dict], keyword_hits: list[dict], limit: int) -> list[dict]:
    """RRF over ranks: fused_score = sum(1 / (RRF_K + rank)) across the routes that found the chunk.

    Hits are dicts with chunk_id and score, best first. Order: fused_score desc, then
    semantic_rank (missing last), then chunk_id. fused_rank is 1-based over the full fused list.
    """
    table: dict[str, dict] = {}

    def entry(chunk_id: str) -> dict:
        return table.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "semantic_score": None, "semantic_rank": None,
                "keyword_score": None, "keyword_rank": None,
            },
        )

    for rank, hit in enumerate(semantic_hits, 1):
        entry(hit["chunk_id"]).update(semantic_rank=rank, semantic_score=hit["score"])
    for rank, hit in enumerate(keyword_hits, 1):
        entry(hit["chunk_id"]).update(keyword_rank=rank, keyword_score=hit["score"])
    for item in table.values():
        item["fused_score"] = sum(
            1.0 / (RRF_K + item[f"{route}_rank"])
            for route in ("semantic", "keyword")
            if item[f"{route}_rank"] is not None
        )
    ordered = sorted(
        table.values(),
        key=lambda e: (
            -e["fused_score"],
            e["semantic_rank"] if e["semantic_rank"] is not None else float("inf"),
            e["chunk_id"],
        ),
    )
    for position, item in enumerate(ordered, 1):
        item["fused_rank"] = position
    return ordered[:limit]


def _ensure_ready(db) -> None:
    """Credentials, both search indexes queryable, embeddings non-empty; positive result cached."""
    global _ready
    if _ready:
        return
    semantic._voyage_client()
    try:
        vector_index = next(
            iter(db[schema.EMBEDDINGS_COLLECTION].list_search_indexes(schema.VECTOR_INDEX_NAME)),
            None,
        )
        keyword_index = next(
            iter(db[schema.CHUNKS_COLLECTION].list_search_indexes(schema.KEYWORD_INDEX_NAME)),
            None,
        )
        has_embeddings = db[schema.EMBEDDINGS_COLLECTION].find_one({}, {"_id": 1}) is not None
    except PyMongoError:
        raise semantic._upstream("MongoDB") from None
    if vector_index is None or not vector_index.get("queryable"):
        raise semantic._not_ready(
            f"Vector index '{schema.VECTOR_INDEX_NAME}' is missing or not queryable. "
            "Run: uv run python -m building_with_rag.ingestion.ingest"
        )
    if keyword_index is None or not keyword_index.get("queryable"):
        raise semantic._not_ready(
            f"Keyword index '{schema.KEYWORD_INDEX_NAME}' is missing or not queryable. "
            "Run: uv run python -m building_with_rag.ingestion.keyword_index"
        )
    if not has_embeddings:
        raise semantic._not_ready(
            "Embeddings collection is empty. Run: uv run python -m building_with_rag.ingestion.ingest"
        )
    _ready = True


def keyword_search(db, question: str, filters: dict[str, list[str]], depth: int) -> list[dict]:
    """Atlas Search `text` operator on chunks.text; filters applied inside compound.filter.

    The question is only ever the text.query value, never an operator or pipeline fragment.
    """
    clauses = [{"in": {"path": field, "value": values}} for field, values in filters.items() if values]
    return list(
        db[schema.CHUNKS_COLLECTION].aggregate(
            [
                {
                    "$search": {
                        "index": schema.KEYWORD_INDEX_NAME,
                        "compound": {
                            "must": [{"text": {"query": question, "path": KEYWORD_PATH}}],
                            "filter": clauses,
                        },
                    }
                },
                {"$limit": depth},
                {
                    "$project": {
                        "_id": 0,
                        "chunk_id": 1,
                        "section_id": 1,
                        "score": {"$meta": "searchScore"},
                    }
                },
            ]
        )
    )


def run_hybrid(request: QueryRequest) -> QueryResult:
    global _ready
    semantic._validate_scope(request, "hybrid")
    limit = request.limit
    depth = route_depth(limit)
    candidates = semantic.num_candidates(depth)
    filters = semantic.effective_filters(request)
    mongo_filter = semantic._mongo_filter(filters)

    trace: dict = {
        "mode": "hybrid",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": semantic.INPUT_TYPE,
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
        "unresolved_hits": 0,
        "semantic": {
            "index": schema.VECTOR_INDEX_NAME,
            "limit": depth,
            "num_candidates": candidates,
            "hit_count": 0,
        },
        "keyword": {
            "index": schema.KEYWORD_INDEX_NAME,
            "path": KEYWORD_PATH,
            "operator": "text",
            "limit": depth,
            "hit_count": 0,
        },
        "fusion": {"method": "rrf", "k": RRF_K, "weights": WEIGHTS, "route_depth": depth},
        "contribution": {"both": 0, "semantic_only": 0, "keyword_only": 0},
    }

    try:
        db = semantic._db()
        _ensure_ready(db)
        try:
            emb = db[schema.EMBEDDINGS_COLLECTION]
            if emb.count_documents(mongo_filter, limit=1) == 0:
                return QueryResult(
                    pattern="hybrid",
                    status="no_results",
                    message="No indexed passages match the given filters.",
                    trace=trace,
                )
            vector = semantic._embed(request.question)
            semantic_hits = semantic.vector_search(emb, vector, depth, candidates, mongo_filter)
            keyword_hits = keyword_search(db, request.question, filters, depth)
            fused = fuse(semantic_hits, keyword_hits, limit)
            chunks, sections = semantic.resolve_chunks(db, [f["chunk_id"] for f in fused])
        except PyMongoError:
            raise semantic._upstream("MongoDB") from None
    except RetrievalError:
        _ready = False  # re-check readiness after any failure
        raise

    trace["semantic"]["hit_count"] = len(semantic_hits)
    trace["keyword"]["hit_count"] = len(keyword_hits)
    results: list[RetrievedChunk] = []
    unresolved = 0
    for item in fused:
        chunk = chunks.get(item["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(
            semantic.build_chunk(
                chunk,
                section,
                item["fused_score"],
                semantic_score=item["semantic_score"],
                semantic_rank=item["semantic_rank"],
                keyword_score=item["keyword_score"],
                keyword_rank=item["keyword_rank"],
                fused_score=item["fused_score"],
                fused_rank=item["fused_rank"],
            )
        )
        found_by = (item["semantic_rank"] is not None, item["keyword_rank"] is not None)
        trace["contribution"][
            {(True, True): "both", (True, False): "semantic_only", (False, True): "keyword_only"}[
                found_by
            ]
        ] += 1
    trace["result_count"] = len(results)
    trace["unresolved_hits"] = unresolved
    note = "The fused score ranks only; it does not prove a passage is correct or answers the question."
    return QueryResult(
        pattern="hybrid",
        status="ok" if results else "no_results",
        message=(
            f"Returned {len(results)} passage(s) fused from semantic and keyword search. {note}"
            if results
            else f"No passages found by either route. {note}"
        ),
        trace=trace,
        results=results,
    )
