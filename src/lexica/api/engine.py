from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import traceback

from lexica.api.config import settings
from lexica.pipeline.orchestration import run_pipeline
from lexica.pipeline.nl_to_ir.validate import IRValidationError


@dataclass
class EngineResult:
    ok: bool
    mode: str  # success | translation_failed | runtime_failed
    artifacts: list[dict]
    error: dict | None = None
    logs: list[str] | None = None


def _repo_root() -> Path:
    # robust: walk upward until we find a folder containing `lexica`
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "lexica").exists():
            return parent
    return Path.cwd().resolve()


def _default_step_output_path() -> Path:
    # export_adapter.py writes this exact file
    return _repo_root() / "lexica_output.step"


def _job_dir(job_id: str) -> Path:
    d = Path(settings.artifacts_dir) / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def copy_step_to_job(job_id: str) -> dict:
    src = _default_step_output_path()
    if not src.exists():
        raise RuntimeError(f"Expected export not found: {src}")

    dst = _job_dir(job_id) / "model.step"
    shutil.copyfile(src, dst)

    return {
        "kind": "step",
        "filename": dst.name,
        "mime": "model/step",
        "url": f"/artifacts/{job_id}/{dst.name}",
    }


class LexicaEngine:
    def run_prompt(self, job_id: str, prompt: str) -> EngineResult:
        raise NotImplementedError


class MockEngine(LexicaEngine):
    def run_prompt(self, job_id: str, prompt: str) -> EngineResult:
        # great for frontend development
        return EngineResult(
            ok=True,
            mode="success",
            artifacts=[],
            logs=[f"[mock] prompt={prompt}"],
        )


class LLMEngine(LexicaEngine):
    def run_prompt(self, job_id: str, prompt: str) -> EngineResult:
        try:
            run_pipeline(prompt, debug=False)
            artifact = copy_step_to_job(job_id)
            return EngineResult(ok=True, mode="success", artifacts=[artifact], logs=[])

        except IRValidationError as e:
            return EngineResult(
                ok=False,
                mode="translation_failed",
                artifacts=[],
                error={"type": "IRValidationError", "message": str(e)},
                logs=[],
            )

        except Exception as e:
            return EngineResult(
                ok=False,
                mode="runtime_failed",
                artifacts=[],
                error={
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc().splitlines()[-25:],
                },
                logs=[],
            )


class HybridEngine(LexicaEngine):
    """
    For now same as LLMEngine, but later we can add:
    - rule fallback for primitives
    - retry with semantic rewrite
    - forced schema constraints
    """
    def __init__(self):
        self.llm = LLMEngine()

    def run_prompt(self, job_id: str, prompt: str) -> EngineResult:
        return self.llm.run_prompt(job_id, prompt)


def get_engine() -> LexicaEngine:
    if settings.engine == "mock":
        return MockEngine()
    if settings.engine == "llm":
        return LLMEngine()
    return HybridEngine()
