from fastapi import APIRouter

from summarizer_web.models.api import (
    ErrorResponse,
    FinalSummaryResponse,
    NodeDetailResponse,
    NodeTreeResponse,
    SegmentListResponse,
)
from summarizer_web.services.run_views_service import (
    get_final_summary,
    get_node_detail,
    get_node_tree,
    get_segments,
)

router = APIRouter(tags=["run views"])

_RUN_NOT_FOUND = {404: {"model": ErrorResponse, "description": "run_not_found"}}


@router.get("/runs/{run_id}/tree", response_model=NodeTreeResponse, responses=_RUN_NOT_FOUND)
def run_tree(run_id: str) -> NodeTreeResponse:
    return get_node_tree(run_id)


@router.get(
    "/runs/{run_id}/nodes/{node_id}",
    response_model=NodeDetailResponse,
    responses={404: {"model": ErrorResponse, "description": "run_not_found or node_not_found"}},
)
def run_node(run_id: str, node_id: str) -> NodeDetailResponse:
    return get_node_detail(run_id, node_id)


@router.get(
    "/runs/{run_id}/segments", response_model=SegmentListResponse, responses=_RUN_NOT_FOUND
)
def run_segments(run_id: str) -> SegmentListResponse:
    return get_segments(run_id)


@router.get(
    "/runs/{run_id}/summary", response_model=FinalSummaryResponse, responses=_RUN_NOT_FOUND
)
def run_summary(run_id: str) -> FinalSummaryResponse:
    return get_final_summary(run_id)
