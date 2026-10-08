"""OpenAI-compatible chat adapter over the shared pipeline.

Maps the selected rag-<pattern> model to the same QueryRequest / retrieve / answer_events path
as /v1/query, with server-side demo caller_id and generate_answer. Only answer writing streams.
Role/content/stop SSE frames; no custom SSE events. The OpenAI-style error envelope applies
before streaming begins.
"""

import hmac
import json
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from building_with_rag.contracts import (
    ChatCompletionRequest,
    GenerationResult,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from building_with_rag.pipeline import answer_events, retrieve, wants_answer
from building_with_rag.registry import MODEL_ID_TO_PATTERN
from building_with_rag.settings import get_settings

router = APIRouter()

_FINISH_STOP = "stop"
_DONE = "[" + "DONE" + "]"
_REASON_LIMIT = 200


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _error(status: int, message: str, err_type: str, code: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": err_type, "code": code}},
    )


def _check_api_key(authorization: str | None) -> None:
    """CAPSTONE_API_KEY set -> require Bearer <key> (constant-time compare); empty -> no check."""
    expected = get_settings().capstone_api_key
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise _error(401, "Invalid API key.", "invalid_request_error", "invalid_api_key")


def _build_request(request: ChatCompletionRequest) -> QueryRequest:
    """Resolve model to pattern and build the shared QueryRequest; raise OpenAI-style errors."""
    pattern = MODEL_ID_TO_PATTERN.get(request.model)
    if pattern is None:
        raise _error(
            400, f"Model '{request.model}' not found.", "invalid_request_error", "model_not_found"
        )
    latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    if latest_user is None:
        raise _error(
            400,
            "At least one user message is required.",
            "invalid_request_error",
            "missing_user_message",
        )
    options = request.rag_options
    nested = options.filters if options else None
    try:
        return QueryRequest(
            question=latest_user,
            pattern=pattern,
            caller_id=get_settings().webui_demo_caller_id,  # server-side demo caller
            filters=(
                SemanticFilters(
                    # Flat fields plus the Pipe's nested rag_options.filters.{act,status}.
                    act=list(dict.fromkeys([*options.act, *(nested.act if nested else [])])),
                    status=list(
                        dict.fromkeys([*options.status, *(nested.status if nested else [])])
                    ),
                    access_level=options.access_level,
                )
                if options
                else None
            ),
            limit=options.limit if options else 5,
            generate_answer=True,  # server-side decision; adapter never trusts client
            required_acts=options.required_acts if options else None,
            chapter=options.chapter if options else None,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        raise _error(
            400, f"Invalid request: {first['msg']}", "invalid_request_error", "invalid_request"
        ) from None


def _retrieve_for_chat(query_request: QueryRequest) -> QueryResult:
    """Run retrieval before any streaming starts; map failures to the OpenAI-style envelope."""
    try:
        return retrieve(query_request)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            raise _error(
                exc.status_code, detail["message"], "server_error", detail["code"]
            ) from None
        message = detail[0]["msg"] if isinstance(detail, list) and detail else "Invalid request."
        raise _error(exc.status_code, message, "invalid_request_error", "invalid_request") from None


def _source_line(citation) -> str:
    act = citation.act.split("_")[0]
    number = citation.section_number or citation.section_id.partition(":")[2]
    return f"{citation.label} · {act} §{number} · {citation.heading} · {citation.section_id}"


def _closing(generation: GenerationResult, streamed_text: bool) -> str:
    """Footer/closing text derived from the same final result the stream described."""
    if generation.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in generation.citations)
        return f"\n\nEvidence check passed — confidence: high\n\nSources:\n{sources}"
    if generation.confidence == "low":
        last = max((i.attempt for i in generation.issues), default=0)
        details = "\n".join(
            f"- {i.check}: {i.detail}" for i in generation.issues if i.attempt == last
        )
        return (
            "\n\nDRAFT — low confidence, not the final answer.\n"
            f"{generation.low_confidence_reason}\n{details}"
        )
    if generation.outcome == "insufficient_evidence":
        reason = str(generation.trace.get("reason", ""))[:_REASON_LIMIT]
        return f"The retrieved passages do not support an answer. {reason}".strip()
    if generation.outcome == "unavailable":
        if streamed_text:
            return "\n\nAnswer generation unavailable — the text above is an unchecked draft."
        return "Answer generation unavailable."
    if streamed_text:
        return "\n\nThe answer could not be checked; treat the text above as an unchecked draft."
    return "The model reply was invalid; no answer is available."


def _pieces(query_request: QueryRequest, retrieval: QueryResult) -> Iterator[str]:
    """Readable text pieces for one operation; stream and JSON responses both use these."""
    if not wants_answer(query_request):
        yield retrieval.message  # placeholder modes keep their existing message
        return
    streamed_text = False
    for event in answer_events(query_request.question, retrieval):
        if event.kind in ("text", "notice"):
            streamed_text = streamed_text or event.kind == "text"
            yield event.text
        else:
            yield _closing(event.generation, streamed_text)


def _chunk(base: dict, n: int, delta: dict, finish: str | None) -> str:
    """One chat.completion.chunk frame; the same piece goes to every choice index."""
    frame = {
        **base,
        "object": "chat.completion.chunk",
        "choices": [{"index": i, "delta": delta, "finish_reason": finish} for i in range(n)],
    }
    return f"data: {json.dumps(frame)}\n\n"


@router.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(default=None)
):
    _check_api_key(authorization)
    query_request = _build_request(request)
    retrieval = _retrieve_for_chat(query_request)

    if not request.stream:
        text = "".join(_pieces(query_request, retrieval))
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _FINISH_STOP,
                }
                for index in range(request.n)
            ],
        }

    def generate():
        base = {"id": _completion_id(), "created": int(time.time()), "model": request.model}
        yield _chunk(base, request.n, {"role": "assistant"}, None)
        for piece in _pieces(query_request, retrieval):
            if piece:
                yield _chunk(base, request.n, {"content": piece}, None)
        yield _chunk(base, request.n, {}, _FINISH_STOP)
        yield "data: " + _DONE + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
