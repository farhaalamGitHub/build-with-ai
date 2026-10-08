"""Story 5.1 offline tests: rule-based classifier and exact lookup against a fake `sections`."""

import pytest

import building_with_rag.settings as settings_module
from building_with_rag import pipeline
from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.context import build_context
from building_with_rag.retrieval import semantic, structured
from building_with_rag.retrieval.structured import classify, run_structured


@pytest.mark.parametrize(
    "question,act,number",
    [
        ("What does BNS section 103 say?", "BNS_2023", 103),
        ("IPC sec. 302", "IPC_1860", 302),
        ("Indian Penal Code section 420", "IPC_1860", 420),
    ],
)
def test_exact_references_are_ok(question, act, number) -> None:
    signals = classify(question)
    assert (signals.status, signals.intent, signals.act, signals.section_number) == (
        "ok", "exact_lookup", act, number)


@pytest.mark.parametrize(
    "question",
    [
        "What does section 103 say?",  # no act
        "BNS and IPC section 103",  # both acts
        "BNS section 103 and 104",  # two numbers
        "BNS section 103A",  # not a plain integer
    ],
)
def test_ambiguous_references_need_clarification(question) -> None:
    signals = classify(question)
    assert signals.status == "clarification_needed" and signals.act is None
    assert signals.section_number is None and signals.reason


def test_aggregation_filter_and_open_questions_are_recommendations() -> None:
    assert classify("How many sections are in the BNS?").intent == "aggregation"
    assert classify("List all sections in chapter V of BNS").intent == "filter"
    open_question = classify("What is the punishment for theft?")
    assert (open_question.status, open_question.intent) == ("recommendation", None)
    assert "semantic" in open_question.reason


class _Sections:
    def __init__(self, record: dict | None) -> None:
        self.record, self.calls = record, []

    def find_one(self, predicate, projection=None):
        self.calls.append((predicate, projection))
        return self.record


@pytest.fixture()
def fake(monkeypatch):
    settings = settings_module.Settings(_env_file=None, mongodb_uri="x")
    monkeypatch.setattr(settings_module, "_settings", settings)

    def install(record: dict | None) -> _Sections:
        sections = _Sections(record)
        monkeypatch.setattr(semantic, "_db", lambda: {"sections": sections})
        return sections

    return install


RECORD = {
    "section_id": "bns:103", "act": "BNS_2023", "act_label": "BNS 2023", "status": "in_force",
    "chapter": "VI", "chapter_title": "Offences", "section_number": 103, "heading": "Murder",
    "text": "(1) Whoever commits murder shall be punished.", "source_pdf": "bns.pdf",
    "source_sha256": "abc", "source_status_version": "v1", "needs_review": False,
}


def _request(question: str, **extra) -> QueryRequest:
    return QueryRequest(question=question, pattern="structured", **extra)


def test_predicate_has_only_validated_fields_never_question_text(fake) -> None:
    sections = fake(RECORD)
    result = run_structured(_request('BNS section 103 {"$ne": null}', chapter="VI",
                                     filters={"status": ["in_force"]}))
    (predicate, projection), = sections.calls
    assert predicate == {
        "act": "BNS_2023", "section_number": 103, "access_level": {"$in": ["public"]},
        "status": {"$in": ["in_force"]}, "chapter": "VI",
    }
    assert projection == {"provenance": 0}
    assert result.status == "ok" and result.trace["mongodb_called"] is True
    chunk, = result.results
    assert (chunk.chunk_id, chunk.section_id, chunk.score) == ("bns:103", "bns:103", 1.0)
    assert result.trace["record"] == {
        "section_id": "bns:103", "status": "in_force", "source_status_version": "v1"}
    assert [e["chunk_id"] for e in build_context(result.results).entries] == ["bns:103"]


@pytest.mark.parametrize(
    "question,kwargs",
    [
        ("What does section 103 say?", {}),
        ("How many sections are there?", {}),
        ("What is theft?", {}),
        ("BNS section 103", {"filters": {"act": ["IPC_1860"]}}),  # filter excludes the act
    ],
)
def test_non_ok_signals_make_no_collection_call(fake, question, kwargs) -> None:
    sections = fake(RECORD)
    result = run_structured(_request(question, **kwargs))
    assert result.status != "ok" and result.results == []
    assert sections.calls == [] and result.trace["mongodb_called"] is False


def test_missing_record_is_not_found(fake) -> None:
    sections = fake(None)
    result = run_structured(_request("IPC section 4"))
    assert result.status == "not_found" and result.results == []
    assert result.trace["mongodb_called"] is True and len(sections.calls) == 1
    assert "does not mean the law has no such section" in result.message


def test_invalid_chapter_and_required_acts_are_422(fake) -> None:
    fake(RECORD)
    for kwargs in ({"chapter": "V; drop"}, {"required_acts": ["BNS_2023"]}):
        with pytest.raises(Exception) as caught:
            run_structured(_request("BNS section 103", **kwargs))
        assert caught.value.status_code == 422


def test_pipeline_routes_structured_and_skips_answer_for_non_ok(monkeypatch) -> None:
    sentinel = QueryResult(pattern="structured", status="clarification_needed", message="m", trace={})
    monkeypatch.setattr(pipeline, "run_structured", lambda request: sentinel)
    request = _request("What does section 103 say?", generate_answer=True)
    assert pipeline.retrieve(request) is sentinel
    assert list(pipeline.answer_events("q", sentinel)) == []
    assert not pipeline.should_answer(request, sentinel)
    ok = QueryResult(pattern="structured", status="ok", message="m", trace={})
    assert pipeline.should_answer(request, ok)
    assert structured.classify("BNS section 103").status == "ok"
