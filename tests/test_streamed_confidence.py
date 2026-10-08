"""Story 3.2 offline tests: fake provider, no network."""

import json
import re

import pytest
from fastapi.testclient import TestClient

import building_with_rag.settings as settings_module
from building_with_rag.app import create_app
from building_with_rag.contracts import QueryResult, RetrievedChunk
from building_with_rag.generation import answer as answer_module
from building_with_rag.generation.answer import ProviderError, stream_answer


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{n}", section_id=f"bns:{300 + n}", act="BNS_2023", text=f"text {n}",
            heading=f"Heading {n}", score=0.9, act_label="BNS 2023", section_number=300 + n,
        )
        for n in (1, 2, 3)
    ]


class FakeProvider:
    """Scripted answer attempts; each entry is a list of stream pieces or an Exception tail."""

    def __init__(self, attempts: list[list]) -> None:
        self.attempts = attempts
        self.calls = 0

    def stream(self, messages):
        script = self.attempts[self.calls]
        self.calls += 1
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item

    @staticmethod
    def validate(messages):
        indexes = re.findall(r"^(\d+)\.", messages[1]["content"], re.M)
        return json.dumps(
            {"claims": [{"index": int(i), "supported": True, "reason": ""} for i in indexes]}
        )


@pytest.fixture()
def setup(monkeypatch):
    settings = settings_module.Settings(
        _env_file=None, generation_api_base_url="http://fake/v1", generation_api_key="k",
        capstone_api_key="",
    )
    monkeypatch.setattr(settings_module, "_settings", settings)

    def install(attempts: list[list]) -> FakeProvider:
        fake = FakeProvider(attempts)
        monkeypatch.setattr(answer_module, "_stream_completion", fake.stream)
        monkeypatch.setattr(answer_module, "_validate_call", fake.validate)
        monkeypatch.setattr(
            "building_with_rag.pipeline.semantic_query",
            lambda request: QueryResult(
                pattern="semantic", status="ok", message="Retrieved.", trace={}, results=_chunks()
            ),
        )
        return fake

    return install


def _chat_text(client: TestClient, model: str = "rag-semantic") -> tuple[str, list[str], str]:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "stream": True, "messages": [{"role": "user", "content": "q"}]},
    ) as response:
        raw = b"".join(response.iter_bytes()).decode()
    frames = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(f) for f in frames[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    finishes = [c["choices"][0]["finish_reason"] for c in chunks]
    return text, finishes, frames[-1]


def _query(client: TestClient) -> dict:
    body = {"question": "q", "pattern": "semantic", "generate_answer": True}
    return client.post("/v1/query", json=body).json()


GOOD = ["BNS section 301 punishes theft ", "[E1]."]


def test_passing_attempt_one_call_stream_equals_final(setup) -> None:
    fake = setup([GOOD])
    events = list(stream_answer("q", _chunks()))
    final = events[-1].generation
    streamed = "".join(e.text for e in events if e.kind == "text")
    assert fake.calls == 1
    assert final.outcome == "answered" and final.confidence == "high"
    assert streamed == final.text
    assert [c.section_id for c in final.citations] == ["bns:301"]

    fake = setup([GOOD])
    text, finishes, last = _chat_text(TestClient(create_app()))
    assert fake.calls == 1
    assert text.startswith("DRAFT — checking evidence")
    assert "Evidence check passed — confidence: high" in text
    assert "E1 · BNS §301 · Heading 1 · bns:301" in text
    assert finishes[-1] == "stop" and last == "[DONE]"


def test_failed_attempt_then_passing_retry(setup) -> None:
    fake = setup([["BNS punishes theft."], GOOD])
    client = TestClient(create_app())
    text, _, _ = _chat_text(client)
    assert fake.calls == 2
    assert text.count("DRAFT — checking evidence") == 2
    assert "Retrying (attempt 2 of 2)" in text

    setup([["BNS punishes theft."], GOOD])
    generation = _query(client)["generation"]
    assert generation["confidence"] == "high" and generation["outcome"] == "answered"
    assert [a["status"] for a in generation["attempts"]] == ["failed", "passed"]
    assert [i["check"] for i in generation["issues"]] == ["claim_cited"]


def test_both_attempts_fail_is_low_confidence(setup) -> None:
    bad = ["BNS punishes theft [E9]."]
    setup([bad, bad])
    client = TestClient(create_app())
    text, _, last = _chat_text(client)
    assert "DRAFT — low confidence, not the final answer." in text and last == "[DONE]"

    setup([bad, bad])
    generation = _query(client)["generation"]
    assert generation["outcome"] == "malformed" and generation["confidence"] == "low"
    assert generation["text"] == "" and generation["claims"] == []
    assert "[E9]" in generation["draft_answer"]
    assert generation["low_confidence_reason"]
    assert len(generation["issues"]) == 2 and len(generation["attempts"]) == 2


def test_provider_failure_mid_stream_is_unavailable(setup) -> None:
    setup([["BNS section 301 punishes", ProviderError("timeout")]])
    client = TestClient(create_app())
    text, finishes, last = _chat_text(client)
    assert "Answer generation unavailable — the text above is an unchecked draft." in text
    assert finishes[-1] == "stop" and last == "[DONE]"

    setup([["BNS section 301 punishes", ProviderError("timeout")]])
    generation = _query(client)["generation"]
    assert generation["outcome"] == "unavailable" and generation["confidence"] is None
    assert generation["draft_answer"].startswith("BNS section 301")


def test_insufficient_evidence_sentinel_is_not_streamed(setup) -> None:
    setup([["INSUFFICIENT_", "EVIDENCE: nothing relevant."]])
    text, _, last = _chat_text(TestClient(create_app()))
    assert "DRAFT" not in text and "confidence" not in text
    assert text.startswith("The retrieved passages do not support an answer.") and last == "[DONE]"


def test_capstone_api_key_enforced_when_set(setup, monkeypatch) -> None:
    setup([GOOD])
    monkeypatch.setattr(settings_module.get_settings(), "capstone_api_key", "secret")
    client = TestClient(create_app())
    body = {"model": "rag-structured", "messages": [{"role": "user", "content": "q"}]}
    wrong = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer x"})
    assert wrong.status_code == 401
    assert wrong.json()["detail"]["error"]["code"] == "invalid_api_key"
    right = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer secret"}
    )
    assert right.status_code == 200
