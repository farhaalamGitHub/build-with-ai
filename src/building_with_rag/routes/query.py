"""Query endpoint: semantic is real; every other mode goes through run_pattern."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.generation.answer import generate_answer
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import semantic_query

router = APIRouter()

_OUTCOME_SENTENCES = {
    "answered": "A grounded answer with citations is attached.",
    "insufficient_evidence": "The retrieved passages do not support an answer.",
    "unavailable": "Answer generation is unavailable; retrieved passages are still returned.",
    "malformed": "The model reply was invalid and discarded; retrieved passages are still returned.",
}


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        result = semantic_query(request)
        if request.generate_answer:
            result.generation = generate_answer(request.question, result.results)
            result.message = f"{result.message} {_OUTCOME_SENTENCES[result.generation.outcome]}"
        return result
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
