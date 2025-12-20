from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

# -----------------------------
# Protocols 
# -----------------------------

class NLToIRStage(Protocol):
    def run(self, text: str) -> Any:
        ...


class IRToIRLStage(Protocol):
    def run(self, ir: Any) -> Any:
        ...


# -----------------------------
# Orchestration results
# -----------------------------

@dataclass
class OrchestrationResult:
    nl_input: str
    ir: Any
    irl: Any


# -----------------------------
# Orchestrator
# -----------------------------

class LexicaOrchestrator:
    """
    orchestrator for Lexica.

    Responsibilities:
    - Wire pipeline stages
    - Enforce execution order
    - Provide clean entrypoint for NL -> IR -> IRL

    This class must NEVER contain domain logic.
    """

    def __init__(
        self,
        nl_to_ir: NLToIRStage,
        ir_to_irl: IRToIRLStage,
        *,
        strict: bool = True,
    ):
        self._nl_to_ir = nl_to_ir
        self._ir_to_irl = ir_to_irl
        self._strict = strict

    # -------------------------
    # Public API
    # -------------------------

    def run(self, text: str) -> OrchestrationResult:
        """
        Execute the full Lexica pipeline:
        Natural Language -> IR -> IRL
        """

        self._validate_input(text)

        ir = self._run_nl_to_ir(text)
        irl = self._run_ir_to_irl(ir)

        return OrchestrationResult(
            nl_input=text,
            ir=ir,
            irl=irl,
        )

    # -------------------------
    # Stage execution
    # -------------------------

    def _run_nl_to_ir(self, text: str) -> Any:
        try:
            ir = self._nl_to_ir.run(text)
        except Exception as exc:
            raise PipelineStageError(
                stage="NL → IR",
                original=exc,
            ) from exc

        if self._strict and ir is None:
            raise InvalidStageOutputError("NL → IR returned None")

        return ir

    def _run_ir_to_irl(self, ir: Any) -> Any:
        try:
            irl = self._ir_to_irl.run(ir)
        except Exception as exc:
            raise PipelineStageError(
                stage="IR -> IRL",
                original=exc,
            ) from exc

        if self._strict and irl is None:
            raise InvalidStageOutputError("IR -> IRL returned None")

        return irl

    # -------------------------
    # Validation
    # -------------------------

    def _validate_input(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if self._strict and not text.strip():
            raise ValueError("Input text cannot be empty")


# -----------------------------
# Errors
# -----------------------------

class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, original: Exception):
        super().__init__(f"Pipeline failed at stage: {stage}")
        self.stage = stage
        self.original = original


class InvalidStageOutputError(RuntimeError):
    pass
