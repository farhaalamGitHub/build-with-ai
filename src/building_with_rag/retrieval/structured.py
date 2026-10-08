"""Structured exact retrieval: rule-based classifier + one read-only `sections` lookup.

Only validated StructuredSignals (act, section_number, optional chapter) and server filters may
reach MongoDB; the raw question text never enters a predicate. No LLM, no Voyage, no vector or
keyword index, no fallback to or from other modes.
"""

import logging
import re

from fastapi import HTTPException
from pymongo.errors import PyMongoError

from building_with_rag.contracts import QueryRequest, QueryResult, RetrievedChunk, StructuredSignals
from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval import semantic
from building_with_rag.settings import get_settings

logger = logging.getLogger(__name__)

_AGGREGATION = re.compile(r"\bhow\s+many\b|\bcount\b|\btotal\s+number\b", re.IGNORECASE)
_FILTER = re.compile(r"\blist\b|\ball\s+sections\b|\bwhich\s+sections\b|\bsections?\s+(?:in|under)\b", re.IGNORECASE)
_NUMBER = r"\d+[A-Za-z]?"
_SECTION_REF = re.compile(
    rf"(?:\bsections?\b\.?|\bsecs?\b\.?|\bs\.|§)\s*({_NUMBER}(?:\s*(?:,|and|&|or|to)\s*{_NUMBER})*)",
    re.IGNORECASE,
)
_ACT_TOKENS = {
    "BNS_2023": re.compile(r"\bbns\b|\bbharatiya\s+nyaya\s+sanhita\b", re.IGNORECASE),
    "IPC_1860": re.compile(r"\bipc\b|\bindian\s+penal\s+code\b", re.IGNORECASE),
}
_CHAPTER = re.compile(r"[A-Za-z0-9 .\-]{1,40}")
_RECOMMEND_OPEN = (
    "This question is not an exact section reference; use the semantic or hybrid mode for open questions."
)
_NOTE = (
    "Exact record from the supplied corpus. status and source_status_version describe the source "
    "document and do not claim current legal applicability."
)


def classify(question: str, chapter: str | None = None) -> StructuredSignals:
    """Pure, rule-based; acts are never guessed."""

    def signals(status: str, reason: str, intent=None, act=None, number=None) -> StructuredSignals:
        return StructuredSignals(
            intent=intent, act=act, section_number=number, chapter=chapter, status=status,
            reason=reason,
        )

    if _AGGREGATION.search(question):
        return signals(
            "recommendation",
            "Counting or aggregation is recognised but not executed; ask for one exact section instead.",
            intent="aggregation",
        )
    if _FILTER.search(question):
        return signals(
            "recommendation",
            "Listing or filtering sections is recognised but not executed; ask for one exact section.",
            intent="filter",
        )
    tokens = [t for group in _SECTION_REF.findall(question) for t in re.findall(_NUMBER, group)]
    if not tokens:
        return signals("recommendation", _RECOMMEND_OPEN)

    def clarify(reason: str) -> StructuredSignals:
        return signals("clarification_needed", reason, intent="exact_lookup")

    if any(not t.isdigit() for t in tokens):
        return clarify("Only plain integer section numbers are supported (for example 103, not 103A).")
    numbers = sorted({int(t) for t in tokens})
    if len(numbers) > 1:
        return clarify("Several section numbers were given; ask for one section at a time.")
    if not 1 <= numbers[0] <= 999:
        return clarify("The section number must be between 1 and 999.")
    acts = [act for act, pattern in _ACT_TOKENS.items() if pattern.search(question)]
    if not acts:
        return clarify("No act was named; say BNS or IPC (the act is never guessed).")
    if len(acts) > 1:
        return clarify("Both BNS and IPC were named; ask about one act at a time.")
    return signals(
        "ok", f"Exact lookup: {acts[0]} section {numbers[0]}.", intent="exact_lookup",
        act=acts[0], number=numbers[0],
    )


def _unprocessable(message: str, kind: str) -> HTTPException:
    return HTTPException(
        status_code=422, detail=[{"loc": ["body"], "msg": message, "type": kind}]
    )


def _check_scope(request: QueryRequest) -> None:
    if request.required_acts is not None:
        raise _unprocessable("required_acts not supported in structured mode; remove it.", "unsupported_option")
    if request.caller_id is not None and request.caller_id != get_settings().webui_demo_caller_id:
        raise _unprocessable("caller_id is not the configured demo caller.", "value_error")
    if request.chapter is not None and not _CHAPTER.fullmatch(request.chapter):
        raise _unprocessable(
            "chapter must be 1-40 characters of letters, digits, spaces, '.' and '-'.",
            "unsupported_option",
        )


def _result(status: str, message: str, trace: dict, results=None) -> QueryResult:
    trace["result_count"] = len(results or [])
    return QueryResult(
        pattern="structured", status=status, message=message, trace=trace, results=results or []
    )


def run_structured(request: QueryRequest) -> QueryResult:
    _check_scope(request)
    signals = classify(request.question, request.chapter)
    filters = semantic.effective_filters(request)
    trace: dict = {
        "mode": "structured",
        "signals": signals.model_dump(),
        "mongodb_called": False,
        "collection": schema.SECTIONS_COLLECTION,
        "filters": filters,
        "caller_id": get_settings().webui_demo_caller_id,
        "result_count": 0,
    }
    if signals.status == "ok" and filters["act"] and signals.act not in filters["act"]:
        signals = signals.model_copy(update={
            "status": "clarification_needed",
            "reason": f"The question names {signals.act}, which the act filter excludes.",
        })
        trace["signals"] = signals.model_dump()
    if signals.status != "ok":
        return _result(signals.status, signals.reason, trace)

    # Predicate: validated classifier output and server filters only, never question text.
    predicate: dict = {
        "act": signals.act,
        "section_number": signals.section_number,
        "access_level": {"$in": filters["access_level"]},
    }
    if filters["status"]:
        predicate["status"] = {"$in": filters["status"]}
    if signals.chapter:
        predicate["chapter"] = signals.chapter

    db = semantic._db()  # 503 only when a lookup is actually needed
    trace["mongodb_called"] = True
    try:
        section = db[schema.SECTIONS_COLLECTION].find_one(predicate, {"provenance": 0})
    except PyMongoError as exc:
        logger.warning("structured lookup failed: %s", type(exc).__name__)
        raise semantic._upstream("MongoDB") from None
    if section is None:
        return _result(
            "not_found",
            f"The supplied corpus has no record for {signals.act} section {signals.section_number} "
            "with these filters; this does not mean the law has no such section.",
            trace,
        )
    chunk = RetrievedChunk(
        chunk_id=section["section_id"],  # exact-lookup records have no embedding chunk
        section_id=section["section_id"],
        act=section["act"],
        text=section["text"],
        heading=section.get("heading") or "",
        score=1.0,  # marks an exact match, not a similarity
        act_label=section.get("act_label"),
        status=section.get("status"),
        chapter=section.get("chapter"),
        chapter_title=section.get("chapter_title"),
        section_number=section.get("section_number"),
        source_pdf=section.get("source_pdf"),
        source_sha256=section.get("source_sha256"),
        needs_review=section.get("needs_review"),
    )
    trace["record"] = {
        "section_id": section["section_id"],
        "status": section.get("status"),
        "source_status_version": section.get("source_status_version"),
    }
    return _result("ok", _NOTE, trace, [chunk])
