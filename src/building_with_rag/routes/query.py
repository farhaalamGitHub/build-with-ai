"""Query endpoint: semantic is real; every other mode goes through run_pattern."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.registry import Pattern, run_pattern
from building_with_rag.retrieval.semantic import semantic_query

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    if request.pattern is Pattern.SEMANTIC:
        return semantic_query(request)
    payload = run_pattern(request.pattern, request.question, request.caller_id)
    return QueryResult(**payload)
