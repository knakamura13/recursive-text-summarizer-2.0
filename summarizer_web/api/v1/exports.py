from fastapi import APIRouter
from fastapi.responses import Response

from summarizer_web.models.api import ErrorResponse
from summarizer_web.services.run_views_service import ExportFormat, export_summary

router = APIRouter(prefix="/runs", tags=["exports"])


@router.get(
    "/{run_id}/export/{export_format}",
    response_class=Response,
    responses={
        200: {
            "content": {"text/plain": {}, "text/markdown": {}, "application/json": {}},
            "description": "The summary (txt, md) or its audit (json) as an attachment.",
        },
        404: {"model": ErrorResponse, "description": "run_not_found or export_unavailable"},
    },
)
def export_run(run_id: str, export_format: ExportFormat) -> Response:
    export = export_summary(run_id, export_format)
    return Response(
        content=export.content,
        media_type=export.media_type,
        headers={"Content-Disposition": export.content_disposition},
    )
