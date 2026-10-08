"""Story 3.1/3.2 claim-check tests: no network."""

from building_with_rag.contracts import RetrievedChunk
from building_with_rag.generation.answer import extract_claims, resolve, structural_issues
from building_with_rag.generation.context import build_context


def _chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{n}", section_id=f"bns:{n}", act="BNS_2023", text=f"text {n}",
        heading=f"h{n}", score=0.9 - n / 100,
    )


def _checks(text: str, supplied: set[str]) -> list[str]:
    return [check for check, _ in structural_issues(text, extract_claims(text), supplied)]


def test_unknown_label_is_flagged_not_stripped() -> None:
    text = "BNS punishes theft [E9]."
    assert _checks(text, {"E1", "E2"}) == ["citation_labels"]
    assert "[E9]" in text  # citations are never rewritten


def test_uncited_statement_is_flagged() -> None:
    assert _checks("BNS punishes theft.", {"E1"}) == ["claim_cited"]
    assert _checks("", {"E1"}) == ["claim_cited"]


def test_label_after_period_and_bullets_are_cited() -> None:
    text = "BNS punishes theft. [E1]\n- Section 303 applies. [E2]"
    assert _checks(text, {"E1", "E2"}) == []
    assert [c["labels"] for c in extract_claims(text)] == [["E1"], ["E2"]]


def test_valid_answer_resolves_citations_in_first_cited_order() -> None:
    results = [_chunk(1), _chunk(2), _chunk(3)]
    ctx = build_context(results)
    claims = extract_claims("BNS says A [E2]. BNS also says B [E1, E2].")
    citations, passages = resolve(claims, ctx, results)
    assert [c.label for c in citations] == ["E2", "E1"]
    assert [c.section_id for c in citations] == ["bns:2", "bns:1"]
    assert [p.chunk_id for p in passages] == ["c2", "c1"]
