from pydantic import BaseModel, Field
from typing import Any, Optional


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "lexica-api"


class SubmitJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    kind: str = "step"
    filename: str
    url: str
    mime: str = "model/step"


class JobStatusResponse(BaseModel):
    ok: bool = True
    job_id: str
    status: str  # queued|running|done|failed
    progress: float = 0.0
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
