"""Story 4.1 offline tests: fusion rule and run_hybrid with faked searches, no network."""

import pytest

import building_with_rag.settings as settings_module
from building_with_rag.contracts import QueryRequest
from building_with_rag.retrieval import hybrid, semantic
from building_with_rag.retrieval.hybrid import RRF_K, fuse
from building_with_rag.retrieval.semantic import RetrievalError


def _hits(*ids: str) -> list[dict]:
    return [{"chunk_id": i, "score": 1.0 - n / 10} for n, i in enumerate(ids)]


def test_chunk_in_both_routes_ranks_first_with_rrf_score() -> None:
    fused = fuse(_hits("a", "b"), _hits("c", "b"), limit=5)
    assert [f["chunk_id"] for f in fused][0] == "b"
    top = fused[0]
    assert (top["semantic_rank"], top["keyword_rank"]) == (2, 2)
    assert top["fused_score"] == pytest.approx(2 / (RRF_K + 2))
    assert [f["fused_rank"] for f in fused] == [1, 2, 3]


def test_single_route_chunks_leave_other_route_none() -> None:
    by_id = {f["chunk_id"]: f for f in fuse(_hits("a"), _hits("c"), limit=5)}
    assert by_id["a"]["keyword_rank"] is None and by_id["a"]["keyword_score"] is None
    assert by_id["c"]["semantic_rank"] is None and by_id["c"]["semantic_score"] is None
    assert by_id["a"]["fused_score"] == pytest.approx(1 / (RRF_K + 1))


def test_ties_are_deterministic_semantic_rank_then_chunk_id() -> None:
    # Same fused score: semantic-only rank 1 vs keyword-only rank 1 -> semantic route first.
    assert [f["chunk_id"] for f in fuse(_hits("z"), _hits("a"), limit=5)] == ["z", "a"]
    # Same fused score and no semantic rank: chunk_id decides.
    assert [f["chunk_id"] for f in fuse([], _hits("m", "b"), limit=1)] == ["m"]
    assert [f["chunk_id"] for f in fuse(_hits("b", "a"), [], limit=5)] == ["b", "a"]


class _Coll:
    def __init__(self, indexes: list[dict]) -> None:
        self.indexes = indexes

    def list_search_indexes(self, name=None):
        return [i for i in self.indexes if name is None or i["name"] == name]

    def find_one(self, *args, **kwargs):
        return {"_id": 1}

    def count_documents(self, *args, **kwargs):
        return 1


class _Db:
    def __init__(self, keyword_indexed: bool) -> None:
        vector = [{"name": "vector_index", "queryable": True}]
        keyword = [{"name": "chunk_text_index", "queryable": True}] if keyword_indexed else []
        self.collections = {"embeddings": _Coll(vector), "chunks": _Coll(keyword)}

    def __getitem__(self, name: str) -> _Coll:
        return self.collections[name]


@pytest.fixture()
def fake(monkeypatch):
    settings = settings_module.Settings(_env_file=None, mongodb_uri="x", voyage_api_key="y")
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(hybrid, "_ready", False)
    monkeypatch.setattr(semantic, "_voyage_client", lambda: None)

    def install(keyword_indexed: bool = True) -> None:
        monkeypatch.setattr(semantic, "_db", lambda: _Db(keyword_indexed))
        monkeypatch.setattr(semantic, "_embed", lambda question: [0.0] * 1024)
        monkeypatch.setattr(
            semantic, "vector_search", lambda *args, **kwargs: _hits("c1", "c2")
        )
        monkeypatch.setattr(hybrid, "keyword_search", lambda *args, **kwargs: _hits("c2", "c3"))
        chunk = lambda n: {"chunk_id": f"c{n}", "section_id": f"bns:{n}", "text": f"text {n}"}  # noqa: E731
        section = lambda n: {"section_id": f"bns:{n}", "act": "BNS_2023", "heading": f"h{n}"}  # noqa: E731
        monkeypatch.setattr(
            semantic,
            "resolve_chunks",
            lambda db, ids: ({f"c{n}": chunk(n) for n in (1, 2, 3)},
                             {f"bns:{n}": section(n) for n in (1, 2, 3)}),
        )

    return install


def test_run_hybrid_sets_fused_fields_and_trace(fake) -> None:
    fake()
    result = hybrid.run_hybrid(QueryRequest(question="theft", pattern="hybrid", limit=3))
    assert result.status == "ok" and result.pattern == "hybrid"
    assert [r.chunk_id for r in result.results][0] == "c2"  # found by both routes
    for position, r in enumerate(result.results, 1):
        assert r.score == r.fused_score and r.fused_rank == position
    assert result.results[0].semantic_rank == 2 and result.results[0].keyword_rank == 1
    assert result.results[1].keyword_rank is None or result.results[1].semantic_rank is None
    assert result.trace["contribution"] == {"both": 1, "semantic_only": 1, "keyword_only": 1}
    assert result.trace["fusion"]["k"] == RRF_K and result.trace["keyword"]["operator"] == "text"


def test_missing_keyword_index_is_not_ready_not_no_results(fake) -> None:
    fake(keyword_indexed=False)
    with pytest.raises(RetrievalError) as caught:
        hybrid.run_hybrid(QueryRequest(question="theft", pattern="hybrid"))
    assert caught.value.status_code == 503
    assert caught.value.detail["code"] == "retrieval_not_ready"
    assert "keyword_index" in caught.value.detail["message"]
