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
from lexica.irl.contract import ExportOp, IRLOpCategory
from lexica.pipeline.nl_to_ir.normalize import normalize_ir



class OrchestrationError(Exception):
    pass


def run_pipeline(
    prompt: str,
    *,
    debug: bool = True,
):
    """
    Full Lexica pipeline:

        NL  -->  IR  -->  IRL  -->  CAD  -->  STEP

    Responsibilities:
    - Orchestrate compiler stages
    - Apply normalization & validation
    - Inject MVP defaults (export)
    - Trigger kernel execution

    Returns:
        None (side effect: STEP file is written)
    """

    # --------------------------------------------------
    # Stage 1: NL --> IR
    # --------------------------------------------------
    ir_model = nl_to_ir(prompt)

    if debug:
        print("[orchestration] IR model (raw):")
        print(ir_model)

    # Normalize LLM-produced IR (semantic completion)
    ir_model = normalize_ir(ir_model)

    if debug:
        print("[orchestration] IR model (normalized):")
        print(ir_model)

    # Strict IR validation
    validate_ir(ir_model)

    # --------------------------------------------------
    # Stage 2: IR --> IRL
    # --------------------------------------------------
    irl_model = lower_ir_to_irl(ir_model)

    if debug:
        print("[orchestration] IRL model (before export):")
        for op in irl_model.ops:
            print(op)

    # Strict IRL validation
    validate_irl(irl_model)

    # --------------------------------------------------
    # Stage 2.5: MVP default export injection
    # --------------------------------------------------
    if not irl_model.ops:
        raise OrchestrationError("IRL model contains no operations")

    last_op = irl_model.ops[-1]

    irl_model.ops.append(
        ExportOp(
            op_id="export_0",
            reads=[last_op.writes],
            writes=None,
            params={"format": "step"},
            category=IRLOpCategory.EXPORT,
        )
    )

    if debug:
        print("[orchestration] IRL model (after export injection):")
        for op in irl_model.ops:
            print(op)

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
    debug: bool = True,
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
    
    last_op = irl_model.ops[-1]
    
    irl_model.ops.append(
        ExportOp(
            op_id="export_0",
            reads=[last_op.writes],
            writes=None,
            params={"format": "step"},
            category=IRLOpCategory.EXPORT,
        )
    )

    if debug:
        print("[orchestration] IRL model:")
        for op in irl_model.ops:
            print(op)

    validate_irl(irl_model)

    executor = IRLExecutor()
    executor.execute(irl_model)

    if debug:
        print("[orchestration] IRL execution completed")

