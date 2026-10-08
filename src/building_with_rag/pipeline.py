"""The one shared path for /v1/query and /v1/chat/completions.

retrieve -> (context assembly -> generation -> validation) -> final outcome.
Retrieval and context assembly are never streamed; only answer writing yields events.
"""

from collections.abc import Iterator

from building_with_rag.contracts import GenerationResult, QueryRequest, QueryResult
from building_with_rag.generation.answer import Event, stream_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.hybrid import run_hybrid
from building_with_rag.retrieval.rerank import run_hybrid_reranked
from building_with_rag.retrieval.semantic import semantic_query
from building_with_rag.retrieval.structured import run_structured

OUTCOME_SENTENCES = {
    "answered": "A grounded answer with citations is attached.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and discarded; retrieved passages are still returned.",
}
LOW_CONFIDENCE_SENTENCE = (
    "The answer failed evidence checks (low confidence) and is withheld; "
    "see generation.issues and generation.draft_answer."
)


# Modes with real retrieval; each shares context assembly, generation, and streaming.
REAL_PATTERNS = frozenset(
    {Pattern.SEMANTIC, Pattern.HYBRID, Pattern.HYBRID_RERANKED, Pattern.STRUCTURED}
)


def retrieve(request: QueryRequest) -> QueryResult:
    """The selected mode decides the route (no automatic routing); others -> placeholder."""
    if request.pattern is Pattern.SEMANTIC:
        return semantic_query(request)
    if request.pattern is Pattern.HYBRID:
        return run_hybrid(request)
    if request.pattern is Pattern.HYBRID_RERANKED:
        return run_hybrid_reranked(request)
    if request.pattern is Pattern.STRUCTURED:
        return run_structured(request)
    return QueryResult(**run_pattern(request.pattern, request.question, request.caller_id))


def answer_events(question: str, retrieval: QueryResult) -> Iterator[Event]:
    """Run the single generation/validation operation; the last event carries the final result."""
    if retrieval.pattern == Pattern.STRUCTURED.value and retrieval.status != "ok":
        return iter(())  # clarification/recommendation/not_found never produce an answer
    return stream_answer(question, retrieval.results)


def attach_generation(retrieval: QueryResult, generation: GenerationResult) -> QueryResult:
    retrieval.generation = generation
    sentence = (
        LOW_CONFIDENCE_SENTENCE
        if generation.confidence == "low"
        else OUTCOME_SENTENCES[generation.outcome]
    )
    retrieval.message = f"{retrieval.message} {sentence}"
    return retrieval


def wants_answer(request: QueryRequest) -> bool:
    return request.pattern in REAL_PATTERNS and request.generate_answer


def should_answer(request: QueryRequest, retrieval: QueryResult) -> bool:
    """Answer only when requested, and for structured only after an exact `ok` lookup."""
    if not wants_answer(request):
        return False
    return request.pattern is not Pattern.STRUCTURED or retrieval.status == "ok"


def run_query(request: QueryRequest) -> QueryResult:
    """Non-streaming caller (/v1/query): retrieve, drain the events, attach the final result."""
    retrieval = retrieve(request)
    if should_answer(request, retrieval):
        for event in answer_events(request.question, retrieval):
            if event.kind == "final":
                attach_generation(retrieval, event.generation)
    return retrieval
