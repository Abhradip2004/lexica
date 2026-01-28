from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import threading

from lexica.api.schemas import HealthResponse, SubmitJobRequest, JobStatusResponse
from lexica.api.jobs import job_store
from lexica.api.engine import get_engine
from lexica.api.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(ok=True)


@router.post("/jobs/submit", response_model=JobStatusResponse)
def submit_job(req: SubmitJobRequest):
    job = job_store.create()

    def worker():
        try:
            job_store.update(job.job_id, status="running", progress=0.05, message="Running...")

            engine = get_engine()
            res = engine.run_prompt(job.job_id, req.prompt)

            job_store.update(
                job.job_id,
                status="done",
                progress=1.0,
                message="Done" if res.mode == "success" else (res.error["message"] if res.error else "Failed"),
                result={
                    "mode": res.mode,
                    "artifacts": res.artifacts,
                    "logs": res.logs or [],
                },
                error=res.error,
            )

        except Exception as e:
            job_store.update(
                job.job_id,
                status="failed",
                progress=1.0,
                message="Internal failure",
                error={"type": e.__class__.__name__, "message": str(e)},
            )

    threading.Thread(target=worker, daemon=True).start()

    return JobStatusResponse(
        ok=True,
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    try:
        job = job_store.get(job_id)
        return JobStatusResponse(
            ok=True,
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            result=job.result,
            error=job.error,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail={"type": "NotFound", "message": "Job not found"})


@router.get("/artifacts/{job_id}/{filename}")
def download_artifact(job_id: str, filename: str):
    base = Path(settings.artifacts_dir).resolve()
    target = (base / job_id / filename).resolve()

    # path safety
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail={"type": "BadPath", "message": "Invalid artifact path"})

    if not target.exists():
        raise HTTPException(status_code=404, detail={"type": "NotFound", "message": "Artifact not found"})

    return FileResponse(path=str(target), filename=filename)
