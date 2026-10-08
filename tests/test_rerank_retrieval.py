"""Story 4.2 offline tests: fake re-ranker provider, no network."""

import httpx
import pytest

import building_with_rag.settings as settings_module
from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import rerank
from building_with_rag.retrieval.semantic import RetrievalError


_REAL_CALL = rerank.call_reranker  # captured before fixtures replace it


def _candidates(n: int = 4) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}", section_id=f"bns:{i}", act="BNS_2023", text=f"text {i}",
            heading=f"h{i}", score=1 / (60 + i), semantic_rank=i, fused_rank=i,
            fused_score=1 / (60 + i),
        )
        for i in range(1, n + 1)
    ]


def _request(limit: int = 5) -> QueryRequest:
    return QueryRequest(question="q", pattern="hybrid-reranked", limit=limit)


@pytest.fixture()
def setup(monkeypatch):
    calls = {"hybrid": 0, "provider": 0}

    def install(candidate=4, send=3, ret=2, key="k", scores=None, **overrides):
        settings = settings_module.Settings(
            _env_file=None, rerank_api_key=key, rerank_api_base_url="http://r/v1",
            rerank_model_name="m", rerank_candidate_limit=candidate, rerank_send_limit=send,
            rerank_return_limit=ret, **overrides,
        )
        monkeypatch.setattr(settings_module, "_settings", settings)

        def fake_hybrid(request):
            calls["hybrid"] += 1
            assert request.limit == candidate and request.pattern.value == "hybrid"
            return QueryResult(
                pattern="hybrid", status="ok", message="", trace={"filters": {}, "caller_id": "c"},
                results=_candidates(candidate),
            )

        def fake_provider(query, documents, settings):
            calls["provider"] += 1
            return (scores or {i: 1.0 - i / 10 for i in range(len(documents))}), 7

        monkeypatch.setattr(rerank, "run_hybrid", fake_hybrid)
        monkeypatch.setattr(rerank, "call_reranker", fake_provider)
        return calls

    return install


def test_reranker_reorders_and_keeps_fused_evidence(setup) -> None:
    setup(candidate=4, send=4, ret=4, scores={0: 0.1, 1: 0.9, 2: 0.5, 3: 0.7})
    result = rerank.run_hybrid_reranked(_request())
    assert [r.chunk_id for r in result.results] == ["c2", "c4", "c3", "c1"]
    assert [r.rerank_rank for r in result.results] == [1, 2, 3, 4]
    assert [r.fused_rank for r in result.results] == [2, 4, 3, 1]  # "before" evidence kept
    assert all(r.score == r.rerank_score for r in result.results)
    scores = [r.score for r in result.results]
    assert scores == sorted(scores, reverse=True)
    assert result.omitted_candidates == [] and result.trace["rerank"]["usage_tokens"] == 7


def test_cut_before_and_after_reranking(setup) -> None:
    setup(candidate=4, send=3, ret=2, scores={0: 0.1, 1: 0.9, 2: 0.5})
    result = rerank.run_hybrid_reranked(_request())
    assert [r.chunk_id for r in result.results] == ["c2", "c3"]
    below, not_sent = result.omitted_candidates  # fused_rank order: c1 then c4
    assert (below.chunk_id, below.omitted_reason) == ("c1", "below_return_limit")
    assert (below.rerank_score, below.rerank_rank) == (0.1, 3)
    assert (not_sent.chunk_id, not_sent.omitted_reason) == ("c4", "not_sent_to_reranker")
    assert not_sent.rerank_score is None and not_sent.rerank_rank is None
    counts = result.trace["rerank"]
    assert (counts["candidates"], counts["sent"], counts["returned"]) == (4, 3, 2)
    assert (counts["omitted_before"], counts["omitted_after"]) == (1, 1)
    # Context assembly sees only the final evidence, never omitted candidates.
    assert [e["chunk_id"] for e in build_context(result.results).entries] == ["c2", "c3"]


def test_request_limit_below_return_limit_shrinks_final(setup) -> None:
    setup(candidate=4, send=3, ret=3, scores={0: 0.1, 1: 0.9, 2: 0.5})
    result = rerank.run_hybrid_reranked(_request(limit=1))
    assert [r.chunk_id for r in result.results] == ["c2"]
    assert [c.omitted_reason for c in result.omitted_candidates] == [
        "below_return_limit", "below_return_limit", "not_sent_to_reranker"]


def test_empty_key_is_not_ready_and_calls_nothing(setup) -> None:
    calls = setup(key="")
    with pytest.raises(RetrievalError) as caught:
        rerank.run_hybrid_reranked(_request())
    assert caught.value.status_code == 503
    assert caught.value.detail["code"] == "retrieval_not_ready"
    assert "RERANK_API_KEY" in caught.value.detail["message"]
    assert calls == {"hybrid": 0, "provider": 0}


def test_invalid_limits_are_not_ready(setup) -> None:
    calls = setup(candidate=3, send=4, ret=2)
    with pytest.raises(RetrievalError) as caught:
        rerank.run_hybrid_reranked(_request())
    assert caught.value.status_code == 503 and "RERANK_SEND_LIMIT" in caught.value.detail["message"]
    assert calls["hybrid"] == 0


def _post(monkeypatch, response=None, error=None) -> None:
    def fake_post(url, **kwargs):
        if error:
            raise error
        return response

    monkeypatch.setattr(rerank.httpx, "post", fake_post)


def _reply(status: int, body) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://r/v1/rerank"))


@pytest.mark.parametrize(
    "response,error",
    [
        (None, httpx.ReadTimeout("slow")),
        (_reply(500, {"error": "x"}), None),
        (_reply(200, {"data": [{"index": 9, "relevance_score": 0.5}]}), None),
        (_reply(200, {"data": [{"index": 0, "relevance_score": 0.5},
                               {"index": 0, "relevance_score": 0.4}]}), None),
        (_reply(200, {"data": [{"index": 0}]}), None),
        (_reply(200, {"data": []}), None),
    ],
)
def test_provider_failures_are_502_without_fallback(setup, monkeypatch, response, error) -> None:
    setup(candidate=2, send=2, ret=1)
    monkeypatch.setattr(rerank, "call_reranker", _REAL_CALL)  # exercise the real HTTP layer
    _post(monkeypatch, response, error)
    with pytest.raises(RetrievalError) as caught:
        rerank.run_hybrid_reranked(_request())
    assert caught.value.status_code == 502
    assert caught.value.detail["code"] == "retrieval_upstream_error"
    assert "http" not in caught.value.detail["message"].lower()



def test_pipeline_routes_hybrid_reranked_here(monkeypatch) -> None:
    sentinel = QueryResult(pattern="hybrid-reranked", status="ok", message="m", trace={})
    monkeypatch.setattr(pipeline, "run_hybrid_reranked", lambda request: sentinel)
    assert pipeline.retrieve(_request()) is sentinel
    assert pipeline.wants_answer(
        QueryRequest(question="q", pattern="hybrid-reranked", generate_answer=True)
    )
