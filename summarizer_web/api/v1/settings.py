from fastapi import APIRouter

from summarizer_web.models.api import ErrorResponse, SettingsResponse, SettingsUpdate
from summarizer_web.services.settings_service import get_settings, update_settings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
def read_settings() -> SettingsResponse:
    return get_settings()


@router.patch(
    "",
    response_model=SettingsResponse,
    responses={422: {"model": ErrorResponse, "description": "invalid_request"}},
)
def patch_settings(payload: SettingsUpdate) -> SettingsResponse:
    return update_settings(payload)
