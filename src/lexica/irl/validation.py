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
    TransformOp,
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
    Validate body lifetimes across IRL ops.

    Rules:
    - A body must be written before it is read
    - PRIMITIVE creates a new LIVE body
    - TRANSFORM creates a new LIVE body, source remains LIVE
    - FEATURE (overwrite=True) kills its input body
    - BOOLEAN kills all input bodies
    - EXPORT does not affect lifetimes
    """

    live_bodies: set[str] = set()

    for op in model.ops:
        # --------------------------
        # Validate reads
        # --------------------------
        for body in op.reads:
            if body not in live_bodies:
                raise IRLValidationError(
                    f"Op '{op.op_id}' reads DEAD or unknown body '{body}'"
                )

        # --------------------------
        # Apply op semantics
        # --------------------------

        # PRIMITIVE: create body
        if op.category == IRLOpCategory.PRIMITIVE:
            live_bodies.add(op.writes)

        # TRANSFORM: create new body, keep source alive
        elif op.category == IRLOpCategory.TRANSFORM:
            live_bodies.add(op.writes)

        # FEATURE: overwrite semantics
        elif op.category == IRLOpCategory.FEATURE:
            if op.overwrite:
                for body in op.reads:
                    live_bodies.remove(body)
            live_bodies.add(op.writes)

        # BOOLEAN: consumes all inputs
        elif op.category == IRLOpCategory.BOOLEAN:
            for body in op.reads:
                live_bodies.remove(body)
            live_bodies.add(op.writes)

        # EXPORT: no lifecycle change
        elif op.category == IRLOpCategory.EXPORT:
            pass

        else:
            raise IRLValidationError(
                f"Unknown IRL op category '{op.category}' in body flow validation"
            )



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

            # Topology-independent features
            elif kind in ("shell", "hole"):
                if op.topo is not None:
                    raise IRLValidationError(
                        f"Feature op '{op.op_id}' of kind '{kind}' must not specify topology"
                    )

            else:
                raise IRLValidationError(
                    f"Unknown feature kind '{kind}' in op '{op.op_id}'"
                )

        # --------------------------
        # Transform
        # --------------------------
        elif cat == IRLOpCategory.TRANSFORM:
            if not isinstance(op, TransformOp):
                raise IRLValidationError(
                    f"Transform op '{op.op_id}' has wrong type"
                )

            if len(op.reads) != 1:
                raise IRLValidationError(
                    f"Transform op '{op.op_id}' must read exactly one body"
                )

            if op.writes is None:
                raise IRLValidationError(
                    f"Transform op '{op.op_id}' must write a body"
                )

            if op.topo is not None:
                raise IRLValidationError(
                    f"Transform op '{op.op_id}' must not specify topology"
                )

            kind = op.params.get("kind")
            if kind not in ("translate", "rotate"):
                raise IRLValidationError(
                    f"Transform op '{op.op_id}' has invalid kind '{kind}'"
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

