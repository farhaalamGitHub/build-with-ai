"""Semantic retrieval: embed the question, run MongoDB $vectorSearch, resolve source passages."""

import threading

from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

INPUT_TYPE = "query"
MIN_CANDIDATES = 50
MAX_CANDIDATES = 200
CANDIDATE_MULTIPLIER = 10  # numCandidates = clamp(limit * 10, max(limit, 50), 200)
MONGO_TIMEOUT_MS = 5000
VOYAGE_TIMEOUT_S = 15.0
SCORES_NOTE = (
    "Scores rank similarity only; they do not prove a passage is correct or answers the question."
)

_lock = threading.Lock()
_mongo: MongoClient | None = None
_voyage: VoyageClient | None = None
_ready = False


class RetrievalError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message})


def num_candidates(limit: int) -> int:
    return min(MAX_CANDIDATES, max(limit, MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER))


def _not_ready(message: str) -> RetrievalError:
    return RetrievalError(503, "retrieval_not_ready", message)


def _upstream(service: str) -> RetrievalError:
    return RetrievalError(
        502, "retrieval_upstream_error", f"{service} request failed; retry later."
    )


def _unprocessable(message: str) -> HTTPException:
    return HTTPException(
        status_code=422, detail=[{"loc": ["body"], "msg": message, "type": "value_error"}]
    )


def _db():
    """Lazy, process-wide MongoDB database handle."""
    global _mongo
    settings = get_settings()
    if not settings.mongodb_uri:
        raise _not_ready("MONGODB_URI is empty. Set it in .env and restart.")
    with _lock:
        if _mongo is None:
            _mongo = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                connectTimeoutMS=MONGO_TIMEOUT_MS,
                socketTimeoutMS=MONGO_TIMEOUT_MS * 3,
            )
    name = (
        settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    )
    return _mongo[name]


def _voyage_client() -> VoyageClient:
    global _voyage
    settings = get_settings()
    if not settings.voyage_api_key:
        raise _not_ready("VOYAGE_API_KEY is empty. Set it in .env and restart.")
    with _lock:
        if _voyage is None:
            _voyage = VoyageClient(
                api_key=settings.voyage_api_key, max_retries=0, timeout=VOYAGE_TIMEOUT_S
            )
        return _voyage


def _ensure_ready(db) -> None:
    """Cheap readiness check; a positive result is cached per process."""
    global _ready
    if _ready:
        return
    _voyage_client()
    try:
        emb = db[schema.EMBEDDINGS_COLLECTION]
        index = next(iter(emb.list_search_indexes(schema.VECTOR_INDEX_NAME)), None)
        if index is None:
            raise _not_ready(
                f"Vector index '{schema.VECTOR_INDEX_NAME}' is missing. "
                "Run: uv run python -m building_with_rag.ingestion.ingest"
            )
        if not index.get("queryable"):
            raise _not_ready(
                f"Vector index '{schema.VECTOR_INDEX_NAME}' is not queryable yet; retry shortly."
            )
        sample = emb.find_one({}, {"model": 1, "model_version": 1, "dimensions": 1})
    except PyMongoError:
        raise _upstream("MongoDB") from None
    if sample is None:
        raise _not_ready(
            "Embeddings collection is empty. Run: uv run python -m building_with_rag.ingestion.ingest"
        )
    if (
        sample.get("model") != schema.EMBEDDING_MODEL
        or sample.get("model_version") != schema.EMBEDDING_MODEL_VERSION
        or sample.get("dimensions") != schema.EMBEDDING_DIMENSIONS
    ):
        raise _not_ready(
            f"Stored embeddings do not match {schema.EMBEDDING_MODEL}/{schema.EMBEDDING_DIMENSIONS}. "
            "Re-run ingestion."
        )
    _ready = True


def effective_filters(request: QueryRequest) -> dict[str, list[str]]:
    """Server fixes access_level; caller lists only narrow (empty list = no narrowing)."""
    f = request.filters
    return {
        "act": list(f.act) if f else [],
        "status": list(f.status) if f else [],
        "access_level": [schema.DEFAULT_ACCESS_LEVEL],
    }


def _mongo_filter(filters: dict[str, list[str]]) -> dict:
    return {field: {"$in": values} for field, values in filters.items() if values}


def _validate_scope(request: QueryRequest, mode: str = "semantic") -> None:
    unsupported = [
        name
        for name, value in (("required_acts", request.required_acts), ("chapter", request.chapter))
        if value is not None
    ]
    if unsupported:
        raise _unprocessable(f"{', '.join(unsupported)} not supported in {mode} mode; remove it.")
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        raise _unprocessable("caller_id is not the configured demo caller.")


def _embed(question: str) -> list[float]:
    try:
        result = _voyage_client().embed(
            texts=[question], model=schema.EMBEDDING_MODEL, input_type=INPUT_TYPE
        )
        vector = result.embeddings[0]
    except (VoyageError, OSError, ValueError, IndexError):  # never leak details
        raise _upstream("Voyage") from None
    if len(vector) != schema.EMBEDDING_DIMENSIONS:
        raise _not_ready(
            f"Query embedding has {len(vector)} dimensions; expected {schema.EMBEDDING_DIMENSIONS}."
        )
    return vector


def vector_search(emb, vector: list[float], limit: int, candidates: int, mongo_filter: dict):
    """$vectorSearch on embeddings; filters are applied inside the search stage."""
    return list(
        emb.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": schema.VECTOR_INDEX_NAME,
                        "path": "vector",
                        "queryVector": vector,
                        "numCandidates": candidates,
                        "limit": limit,
                        "filter": mongo_filter,
                    }
                },
                {"$project": {"_id": 0, "chunk_id": 1, "score": {"$meta": "vectorSearchScore"}}},
            ]
        )
    )


def resolve_chunks(db, chunk_ids: list[str]) -> tuple[dict, dict]:
    """Fetch chunks (text) and their sections (heading, chapter, source) for the given ids."""
    chunks = {
        c["chunk_id"]: c
        for c in db[schema.CHUNKS_COLLECTION].find({"chunk_id": {"$in": chunk_ids}})
    }
    section_ids = list({c["section_id"] for c in chunks.values()})
    sections = {
        s["section_id"]: s
        for s in db[schema.SECTIONS_COLLECTION].find({"section_id": {"$in": section_ids}})
    }
    return chunks, sections


def build_chunk(chunk: dict, section: dict, score: float, **extra) -> RetrievedChunk:
    """RetrievedChunk from stored records; missing source fields stay None, never guessed."""
    return RetrievedChunk(
        chunk_id=chunk["chunk_id"],
        section_id=section["section_id"],
        act=section["act"],
        text=chunk["text"],
        heading=section.get("heading") or "",
        score=score,
        chunk_index=chunk.get("chunk_index"),
        act_label=section.get("act_label"),
        status=section.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
        **extra,
    )


def semantic_query(request: QueryRequest) -> QueryResult:
    global _ready
    _validate_scope(request)
    limit = request.limit
    filters = effective_filters(request)
    mongo_filter = _mongo_filter(filters)
    candidates = num_candidates(limit)

    trace: dict = {
        "mode": "semantic",
        "query": request.question,
        "embedding": {
            "model": schema.EMBEDDING_MODEL,
            "input_type": INPUT_TYPE,
            "dimensions": schema.EMBEDDING_DIMENSIONS,
        },
        "index": schema.VECTOR_INDEX_NAME,
        "limit": limit,
        "num_candidates": candidates,
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
        "ignored": [],
        "unresolved_hits": 0,
    }

    try:
        db = _db()
        _ensure_ready(db)
        try:
            emb = db[schema.EMBEDDINGS_COLLECTION]
            if emb.count_documents(mongo_filter, limit=1) == 0:
                return QueryResult(
                    pattern="semantic",
                    status="no_results",
                    message="No indexed passages match the given filters.",
                    trace=trace,
                )
            vector = _embed(request.question)
            hits = vector_search(emb, vector, limit, candidates, mongo_filter)
            chunks, sections = resolve_chunks(db, [h["chunk_id"] for h in hits])
        except PyMongoError:
            raise _upstream("MongoDB") from None
    except RetrievalError:
        _ready = False  # re-check readiness after any failure
        raise

    results: list[RetrievedChunk] = []
    unresolved = 0
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        section = sections.get(chunk["section_id"]) if chunk else None
        if chunk is None or section is None:
            unresolved += 1
            continue
        results.append(build_chunk(chunk, section, hit["score"]))
    trace["result_count"] = len(results)
    trace["unresolved_hits"] = unresolved
    return QueryResult(
        pattern="semantic",
        status="ok" if results else "no_results",
        message=(
            f"Returned {len(results)} passage(s) by semantic similarity. {SCORES_NOTE}"
            if results
            else f"No passages resolved for this query. {SCORES_NOTE}"
        ),
        trace=trace,
        results=results,
    )
