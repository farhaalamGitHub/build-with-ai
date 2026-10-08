"""Query endpoint: the richer inspection view over the shared pipeline."""

from fastapi import APIRouter

from building_with_rag.contracts import QueryRequest, QueryResult
from building_with_rag.pipeline import run_query

router = APIRouter()


@router.post("/v1/query")
def query(request: QueryRequest) -> QueryResult:
    return run_query(request)
