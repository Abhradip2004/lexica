"""
IRL Semantic Validation
======================

Purpose:
- Enforce kernel grade IRL correctness
- Catch semantic errors before kernel execution
- Validate body identity, dataflow, and op legality

This module:
- DOES NOT execute ops
- DOES NOT touch CadQuery/OpenCascade
- DOES NOT mutate IRL
"""

from __future__ import annotations

from typing import Set

from lexica.irl.contract import (
    IRLModel,
    IRLOpCategory,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
)


class IRLValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_irl(model: IRLModel) -> None:
    """
    Validate an IRL program.

    Raises:
        IRLValidationError on any semantic violation.
    """

    _validate_op_ids_unique(model)
    _validate_body_flow(model)
    _validate_ops_semantics(model)


# ---------------------------------------------------------------------------
# Validation stages
# ---------------------------------------------------------------------------

def _validate_op_ids_unique(model: IRLModel) -> None:
    seen = set()
    for op in model.ops:
        if op.op_id in seen:
            raise IRLValidationError(
                f"Duplicate IRL op_id: '{op.op_id}'"
            )
        seen.add(op.op_id)


def _validate_body_flow(model: IRLModel) -> None:
    """
    Enforce body lifecycle semantics:
    - bodies must be LIVE before read
    - overwrite features kill input bodies
    - fork features preserve input bodies
    - booleans consume all operands
    """

    live_bodies: Set[str] = set()

    for op in model.ops:
        # --------------------------
        # Reads must be LIVE
        # --------------------------
        for body in op.reads:
            if body not in live_bodies:
                raise IRLValidationError(
                    f"Op '{op.op_id}' reads DEAD or unknown body '{body}'"
                )

        # --------------------------
        # Primitive
        # --------------------------
        if isinstance(op, PrimitiveOp):
            live_bodies.add(op.writes)

        # --------------------------
        # Feature
        # --------------------------
        elif isinstance(op, FeatureOp):
            src = op.reads[0]

            if op.overwrite:
                # kill source body
                live_bodies.remove(src)

            # new body always becomes live
            live_bodies.add(op.writes)

        # --------------------------
        # Boolean
        # --------------------------
        elif isinstance(op, BooleanOp):
            for body in op.reads:
                live_bodies.remove(body)

            live_bodies.add(op.writes)

        # --------------------------
        # Export
        # --------------------------
        elif isinstance(op, ExportOp):
            pass


def _validate_ops_semantics(model: IRLModel) -> None:
    for op in model.ops:
        cat = op.category

        # --------------------------
        # Primitive
        # --------------------------
        if cat == IRLOpCategory.PRIMITIVE:
            if not isinstance(op, PrimitiveOp):
                raise IRLValidationError(
                    f"Primitive op '{op.op_id}' has wrong type"
                )
            if op.reads:
                raise IRLValidationError(
                    f"Primitive op '{op.op_id}' must not read bodies"
                )
            if op.writes is None:
                raise IRLValidationError(
                    f"Primitive op '{op.op_id}' must write a body"
                )

        # --------------------------
        # Feature
        # --------------------------
        elif cat == IRLOpCategory.FEATURE:
            if not isinstance(op, FeatureOp):
                raise IRLValidationError(
                    f"Feature op '{op.op_id}' has wrong type"
                )

            if len(op.reads) != 1:
                raise IRLValidationError(
                    f"Feature op '{op.op_id}' must read exactly one body"
                )

            if op.writes is None:
                raise IRLValidationError(
                    f"Feature op '{op.op_id}' must write a body"
                )

            kind = op.params.get("kind")

            # Topology-dependent features
            if kind in ("fillet", "chamfer"):
                if op.topo is None:
                    raise IRLValidationError(
                        f"Feature op '{op.op_id}' of kind '{kind}' requires topo selection"
                    )

            # Topology-independent features (transforms)
            elif kind in ("translate", "rotate"):
                # topo must be None or ignored
                pass

            else:
                raise IRLValidationError(
                    f"Unknown feature kind '{kind}' in op '{op.op_id}'"
                )

        # --------------------------
        # Boolean
        # --------------------------
        elif cat == IRLOpCategory.BOOLEAN:
            if not isinstance(op, BooleanOp):
                raise IRLValidationError(
                    f"Boolean op '{op.op_id}' has wrong type"
                )
            if len(op.reads) < 2:
                raise IRLValidationError(
                    f"Boolean op '{op.op_id}' must read >= 2 bodies"
                )
            if op.writes is None:
                raise IRLValidationError(
                    f"Boolean op '{op.op_id}' must write a body"
                )

        # --------------------------
        # Export
        # --------------------------
        elif cat == IRLOpCategory.EXPORT:
            if not isinstance(op, ExportOp):
                raise IRLValidationError(
                    f"Export op '{op.op_id}' has wrong type"
                )
            if len(op.reads) != 1:
                raise IRLValidationError(
                    f"Export op '{op.op_id}' must read exactly one body"
                )
            if op.writes is not None:
                raise IRLValidationError(
                    f"Export op '{op.op_id}' must not write a body"
                )

        else:
            raise IRLValidationError(
                f"Unknown IRL op category: {cat}"
            )
