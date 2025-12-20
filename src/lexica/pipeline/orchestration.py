"""
Lexica Pipeline Orchestration
============================

Purpose:
- Coordinate compiler passes
- Enforce stage separation
- Provide a single entry point from NL → STEP

This module:
- DOES NOT execute CAD logic
- DOES NOT mutate shapes
- DOES NOT contain kernel knowledge
"""

from __future__ import annotations

from lexica.pipeline.nl_to_ir.translator import nl_to_ir
from lexica.pipeline.nl_to_ir.validate import validate_ir

from lexica.irl.ir_to_irl import lower_ir_to_irl
from lexica.irl.validation import validate_irl

from lexica.cad_engine.executor import IRLExecutor


class OrchestrationError(Exception):
    pass


def run_pipeline(
    prompt: str,
    *,
    debug: bool = False,
):
    """
    Full Lexica pipeline:
        NL --> IR --> IRL --> CAD --> STEP

    Returns:
        None (side effect: export ops write files)
    """

    # --------------------------------------------------
    # Stage 1: NL --> IR
    # --------------------------------------------------
    ir_model = nl_to_ir(prompt)

    if debug:
        print("[orchestration] IR model:")
        print(ir_model)

    validate_ir(ir_model)

    # --------------------------------------------------
    # Stage 2: IR --> IRL
    # --------------------------------------------------
    irl_model = lower_ir_to_irl(ir_model)

    if debug:
        print("[orchestration] IRL model:")
        for op in irl_model.ops:
            print(op)

    validate_irl(irl_model)

    # --------------------------------------------------
    # Stage 3: IRL --> CAD kernel
    # --------------------------------------------------
    executor = IRLExecutor()
    executor.execute(irl_model)

    if debug:
        print("[orchestration] Execution completed successfully")

def run_irl_pipeline(
    ir_model,
    *,
    debug: bool = False,
):
    """
    IR-only pipeline:
        IR --> IRL --> CAD --> STEP

    Used for:
    - kernel stress tests
    - compiler-internal testing
    """

    validate_ir(ir_model)

    irl_model = lower_ir_to_irl(ir_model)

    if debug:
        print("[orchestration] IRL model:")
        for op in irl_model.ops:
            print(op)

    validate_irl(irl_model)

    executor = IRLExecutor()
    executor.execute(irl_model)

    if debug:
        print("[orchestration] IRL execution completed")

