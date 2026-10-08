"""Hybrid re-ranking: bounded hybrid candidates re-scored by one provider call.

Flow: hybrid retrieval (top RERANK_CANDIDATE_LIMIT) -> first RERANK_SEND_LIMIT sent to the
re-ranker -> ordered by relevance_score -> first min(limit, RERANK_RETURN_LIMIT) kept as
evidence. Cut candidates go to QueryResult.omitted_candidates. No fallback: any failure is an
error with no results, never hybrid order labelled as re-ranked.
"""

import logging
import math
import time

import httpx

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.registry import Pattern
from building_with_rag.retrieval import semantic
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.semantic import RetrievalError
from building_with_rag.settings import Settings, get_settings

MAX_CANDIDATES = 20
logger = logging.getLogger(__name__)


def validate_settings(settings: Settings) -> None:
    """503 retrieval_not_ready naming the setting (never its value)."""
    for name, value in (
        ("RERANK_API_KEY", settings.rerank_api_key),
        ("RERANK_API_BASE_URL", settings.rerank_api_base_url),
        ("RERANK_MODEL_NAME", settings.rerank_model_name),
    ):
        if not str(value).strip():
            raise semantic._not_ready(f"{name} is not set; hybrid-reranked requires it.")
    if settings.rerank_request_timeout_seconds < 1:
        raise semantic._not_ready("RERANK_REQUEST_TIMEOUT_SECONDS must be at least 1.")
    candidate, send, ret = (
        settings.rerank_candidate_limit,
        settings.rerank_send_limit,
        settings.rerank_return_limit,
    )
    if not 1 <= candidate <= MAX_CANDIDATES:
        raise semantic._not_ready(f"RERANK_CANDIDATE_LIMIT must be between 1 and {MAX_CANDIDATES}.")
    if not 1 <= send <= candidate:
        raise semantic._not_ready("RERANK_SEND_LIMIT must be between 1 and RERANK_CANDIDATE_LIMIT.")
    if not 1 <= ret <= send:
        raise semantic._not_ready("RERANK_RETURN_LIMIT must be between 1 and RERANK_SEND_LIMIT.")


def _failed() -> RetrievalError:
    return RetrievalError(
        502,
        "retrieval_upstream_error",
        "Re-ranking request failed; no re-ranked result returned.",
    )


def _document(chunk: RetrievedChunk) -> str:
    return f"{chunk.heading}\n{chunk.text}"


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def call_reranker(query: str, documents: list[str], settings: Settings) -> tuple[dict, int | None]:
    """One POST {base}/rerank. Returns ({index: relevance_score}, usage_tokens). No retries.

    The question is only the `query` value. Any invalid reply is a provider failure; scores are
    never invented, defaulted, or filled in.
    """
    url = f"{settings.rerank_api_base_url.strip().rstrip('/')}/rerank"
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={"model": settings.rerank_model_name, "query": query, "documents": documents},
            timeout=float(settings.rerank_request_timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("rerank call failed: %s", type(exc).__name__)
        raise _failed() from None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise _failed()
    scores: dict[int, float] = {}
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        score = item.get("relevance_score") if isinstance(item, dict) else None
        valid_index = isinstance(index, int) and not isinstance(index, bool)
        if not valid_index or not 0 <= index < len(documents) or index in scores:
            raise _failed()
        if not _is_number(score):
            raise _failed()
        scores[index] = float(score)
    if len(scores) != len(documents):  # every sent candidate must be scored
        raise _failed()
    usage = body.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return scores, tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


def select(
    candidates: list[RetrievedChunk], send_limit: int, scores: dict[int, float], keep: int
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Pure selection. `candidates` are in fused order; `scores` maps sent index -> score.

    Returns (final evidence by rerank_rank, omitted candidates by fused_rank).
    """
    scored = sorted(
        range(min(send_limit, len(candidates))),
        key=lambda i: (-scores[i], candidates[i].fused_rank or 0),
    )
    ranked = {
        i: candidates[i].model_copy(update={"rerank_score": scores[i], "rerank_rank": rank})
        for rank, i in enumerate(scored, 1)
    }
    final = [
        ranked[i].model_copy(update={"score": scores[i]}) for i in scored[:keep]
    ]
    omitted: dict[int, RetrievedChunk] = {}
    for i in scored[keep:]:
        omitted[i] = ranked[i].model_copy(
            update={"score": scores[i], "omitted_reason": "below_return_limit"}
        )
    for i in range(len(scored), len(candidates)):
        omitted[i] = candidates[i].model_copy(update={"omitted_reason": "not_sent_to_reranker"})
    return final, [omitted[i] for i in sorted(omitted)]


def run_hybrid_reranked(request: QueryRequest) -> QueryResult:
    semantic._validate_scope(request, "hybrid-reranked")
    settings = get_settings()
    validate_settings(settings)  # before any Voyage/MongoDB call
    candidate_limit = settings.rerank_candidate_limit
    send_limit = settings.rerank_send_limit
    return_limit = settings.rerank_return_limit

    hybrid = run_hybrid(
        request.model_copy(update={"pattern": Pattern.HYBRID, "limit": candidate_limit})
    )
    candidates = hybrid.results
    trace: dict = {
        "mode": "hybrid-reranked",
        "query": request.question,
        "filters": hybrid.trace.get("filters"),
        "caller_id": hybrid.trace.get("caller_id"),
        "result_count": 0,
        "hybrid": {
            key: hybrid.trace[key]
            for key in ("embedding", "semantic", "keyword", "fusion", "contribution", "unresolved_hits")
            if key in hybrid.trace
        },
        "rerank": {
            "model": settings.rerank_model_name,
            "candidate_limit": candidate_limit,
            "send_limit": send_limit,
            "return_limit": return_limit,
            "candidates": len(candidates),
            "sent": 0,
            "returned": 0,
            "omitted_before": 0,
            "omitted_after": 0,
            "latency_ms": 0,
        },
    }
    note = "Re-ranking scores rank only; they do not prove a passage is correct or answers the question."
    if not candidates:
        return QueryResult(
            pattern="hybrid-reranked",
            status="no_results",
            message=f"No candidates to re-rank. {note}",
            trace=trace,
        )

    sent = candidates[:send_limit]
    started = time.monotonic()
    scores, tokens = call_reranker(request.question, [_document(c) for c in sent], settings)
    final, omitted = select(candidates, send_limit, scores, min(request.limit, return_limit))

    rerank = trace["rerank"]
    rerank.update(
        sent=len(sent),
        returned=len(final),
        omitted_before=sum(1 for c in omitted if c.omitted_reason == "not_sent_to_reranker"),
        omitted_after=sum(1 for c in omitted if c.omitted_reason == "below_return_limit"),
        latency_ms=round((time.monotonic() - started) * 1000),
    )
    if tokens is not None:
        rerank["usage_tokens"] = tokens
    trace["result_count"] = len(final)
    return QueryResult(
        pattern="hybrid-reranked",
        status="ok",
        message=f"Returned {len(final)} re-ranked passage(s) from {len(candidates)} hybrid candidates. {note}",
        trace=trace,
        results=final,
        omitted_candidates=omitted,
    )
