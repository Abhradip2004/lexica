from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import uuid
import time
import threading


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[dict] = None
    created_at: float = field(default_factory=time.time)


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[dict] = None,
    ) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            return job


job_store = JobStore()
