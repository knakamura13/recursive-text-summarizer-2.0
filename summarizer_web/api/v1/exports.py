from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from summarizer_web.config import load_paths
from summarizer_web.db.connection import get_database

router = APIRouter(prefix="/runs", tags=["exports"])


@router.get("/{run_id}/export/{format}")
def export_run(run_id: str, format: str):
    row = get_database().fetchone("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = load_paths().runs / run_id
    if format == "text":
        path = run_dir / "summary.txt"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Summary not available")
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    if format == "audit-json":
        path = run_dir / "audit.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Audit not available")
        return FileResponse(path, media_type="application/json")
    raise HTTPException(status_code=400, detail="Unsupported export format")
