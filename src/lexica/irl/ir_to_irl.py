"""
IR --> IRL lowering pass
=====================

Purpose:
- Convert high level IR into kernel grade IRL
- Assign explicit body identities
- Make feature semantics explicit
- Eliminate all implicit CAD state

This module:
- DOES NOT execute anything
- DOES NOT touch CadQuery/OpenCascade
- DOES NOT mutate global state
"""

from __future__ import annotations

from typing import List

from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    BooleanKind,
    TopoPredicate,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class LoweringError(Exception):
    pass


class BodyIDGenerator:
    """
    Deterministic body id generator.
    """

    def __init__(self):
        self._counter = 0

    def new(self, hint: str = "body") -> str:
        bid = f"{hint}_{self._counter}"
        self._counter += 1
        return bid


# ---------------------------------------------------------------------------
# Main lowering entry
# ---------------------------------------------------------------------------

def lower_ir_to_irl(ir_model) -> IRLModel:
    """
    Lower IR model to IRL model.

    IR assumptions (important):
    - IR expresses intent, not execution
    - IR may reference bodies implicitly
    """

    body_ids = BodyIDGenerator()
    ops: List = []

    # Tracks the "current" body at IR level ONLY
    # This does NOT survive into IRL
    current_body = None

    for ir_op in ir_model.ops:
        kind = ir_op.kind

        # --------------------------------------------------
        # Primitive creation
        # --------------------------------------------------
        if kind == "primitive":
            body_id = body_ids.new(ir_op.primitive_kind)

            ops.append(
                PrimitiveOp(
                    op_id=ir_op.op_id,
                    reads=[],
                    writes=body_id,
                    params={
                        "kind": ir_op.primitive_kind,
                        **ir_op.params,
                    },
                )
            )

            current_body = body_id
            continue

        # --------------------------------------------------
        # Feature operations (fillet, chamfer, etc.)
        # --------------------------------------------------
        if kind == "feature":
            if current_body is None:
                raise LoweringError(
                    f"Feature '{ir_op.op_id}' has no target body"
                )

            new_body = body_ids.new("feat")

            ops.append(
                FeatureOp(
                    op_id=ir_op.op_id,
                    reads=[current_body],
                    writes=new_body,
                    params={
                        "kind": ir_op.feature_kind,
                        **ir_op.params,
                    },
                    topo=_lower_topology(ir_op),
                )
            )

            current_body = new_body
            continue

        # --------------------------------------------------
        # Boolean operations
        # --------------------------------------------------
        if kind == "boolean":
            if len(ir_op.operands) < 2:
                raise LoweringError("Boolean op requires >= 2 operands")

            result_body = body_ids.new("bool")

            ops.append(
                BooleanOp(
                    op_id=ir_op.op_id,
                    reads=list(ir_op.operands),
                    writes=result_body,
                    kind=BooleanKind(ir_op.boolean_kind),
                )
            )

            current_body = result_body
            continue

        # --------------------------------------------------
        # Export
        # --------------------------------------------------
        if kind == "export":
            if current_body is None:
                raise LoweringError("Export has no body to export")

            ops.append(
                ExportOp(
                    op_id=ir_op.op_id,
                    reads=[current_body],
                    writes=None,
                    params={
                        "format": ir_op.format,
                        "path": ir_op.path,
                    },
                )
            )
            continue

        raise LoweringError(f"Unknown IR op kind: {kind}")

    return IRLModel(ops=ops)


# ---------------------------------------------------------------------------
# Topology lowering (Level 0)
# ---------------------------------------------------------------------------

def _lower_topology(ir_op) -> TopoPredicate:
    """
    Convert IR topology intent into IRL TopoPredicate.
    """

    topo = ir_op.topology

    if topo is None:
        raise LoweringError(
            f"Feature '{ir_op.op_id}' requires topology selection"
        )

    return TopoPredicate(
        target=topo.target,
        rule=topo.rule,
        value=topo.value,
    )
