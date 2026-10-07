"""Story 3.1 parser tests: no network."""

import json

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import parse_model_output, resolve
from building_with_rag.generation.context import build_context


def _chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{n}", section_id=f"bns:{n}", act="BNS_2023", text=f"text {n}",
        heading=f"h{n}", score=0.9 - n / 100,
    )


def _answered(evidence: list[str]) -> str:
    return json.dumps(
        {
            "outcome": "answered",
            "answer": "BNS 303 punishes theft.",
            "claims": [{"text": "Theft is punishable.", "evidence": evidence}],
            "reason": "",
        }
    )


def test_unknown_label_is_malformed() -> None:
    assert parse_model_output(_answered(["E9"]), {"E1", "E2"}) is None


def test_non_json_is_malformed() -> None:
    assert parse_model_output("Theft is punishable.", {"E1"}) is None


def test_answered_without_claims_is_malformed() -> None:
    raw = json.dumps({"outcome": "answered", "answer": "x", "claims": [], "reason": ""})
    assert parse_model_output(raw, {"E1"}) is None


def test_valid_answer_resolves_citations_in_first_cited_order() -> None:
    results = [_chunk(1), _chunk(2), _chunk(3)]
    ctx = build_context(results)
    raw = "```json\n" + _answered(["E2", "E1"]) + "\n```"
    parsed = parse_model_output(raw, set(ctx.labels))
    assert parsed is not None
    citations, passages = resolve(parsed, ctx, results)
    assert [c.label for c in citations] == ["E2", "E1"]
    assert [c.section_id for c in citations] == ["bns:2", "bns:1"]
    assert [p.chunk_id for p in passages] == ["c2", "c1"]
