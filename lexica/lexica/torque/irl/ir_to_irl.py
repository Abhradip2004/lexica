"""
IR --> IRL lowering pass
=====================

Purpose:
- Convert high level IR into kernel     grade IRL
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

from lexica.torque.irl.contract import (
    IRLModel,
    TransformOp,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    BooleanKind,
    TopoPredicate,
)

from lexica.torque.language.ir.schema import IROpKind



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

    IR v0 semantics:
    - IR has NO op_id
    - IR has NO body identity
    - Bodies are implicit and sequential
    - Boolean ops operate on:
        - explicit operand indices (if provided), OR
        - current solid body (LHS)
        - most recent tool body (RHS)
    """

    body_ids = BodyIDGenerator()
    ops: List = []

    # Deterministic list of all created bodies (by index)
    body_history: List[str] = []

    # Main solid being built
    current_body = None

    # Most recently created tool body (for booleans)
    last_tool_body = None

    for idx, ir_op in enumerate(ir_model.ops):
        kind = ir_op.kind
        op_id = f"{kind.value}_{idx}"

        # --------------------------------------------------
        # Primitive
        # --------------------------------------------------
        if kind == IROpKind.PRIMITIVE:
            body_id = body_ids.new(ir_op.primitive_kind.value)

            ops.append(
                PrimitiveOp(
                    op_id=op_id,
                    reads=[],
                    writes=body_id,
                    params={
                        "kind": ir_op.primitive_kind.value,
                        **ir_op.params,
                    },
                )
            )

            # First primitive becomes the main body,
            # subsequent primitives are tool bodies
            if current_body is None:
                current_body = body_id
            else:
                last_tool_body = body_id

            body_history.append(body_id)
            continue

        # --------------------------------------------------
        # Transform (translate / rotate)
        # --------------------------------------------------
        if kind == IROpKind.TRANSFORM:
            if last_tool_body is not None:
                target = last_tool_body
            elif current_body is not None:
                target = current_body
            else:
                raise LoweringError("Transform op has no target body")

            new_body = body_ids.new("xf")

            ops.append(
                TransformOp(
                    op_id=op_id,
                    reads=[target],
                    writes=new_body,
                    params={
                        "kind": ir_op.transform_kind.value,
                        **ir_op.params,
                    },
                )
            )

            # Preserve whether this was a tool or the main body
            if target == current_body:
                current_body = new_body
            else:
                last_tool_body = new_body

            body_history.append(new_body)
            continue

        # --------------------------------------------------
        # Feature (fillet / chamfer / shell / hole)
        # --------------------------------------------------
        if kind == IROpKind.FEATURE:
            if current_body is None:
                raise LoweringError("Feature op has no target body")

            new_body = body_ids.new("feat")

            ops.append(
                FeatureOp(
                    op_id=op_id,
                    reads=[current_body],
                    writes=new_body,
                    params={
                        "kind": ir_op.feature_kind.value,
                        **ir_op.params,
                    },
                    topo=_lower_topology(ir_op),
                )
            )

            current_body = new_body
            body_history.append(new_body)
            continue

        # --------------------------------------------------
        # Boolean
        # --------------------------------------------------
        if kind == IROpKind.BOOLEAN:
            if ir_op.operands is not None:
                # Explicit operand indices
                try:
                    reads = [body_history[i] for i in ir_op.operands]
                except IndexError:
                    raise LoweringError(
                        f"Boolean op '{op_id}' references invalid operand index"
                    )

                if len(reads) < 2:
                    raise LoweringError(
                        f"Boolean op '{op_id}' requires at least 2 operands"
                    )

            else:
                # Backward-compatible implicit behavior
                if current_body is None or last_tool_body is None:
                    raise LoweringError(
                        "Boolean op requires a current body and a tool body"
                    )
                reads = [current_body, last_tool_body]

            result_body = body_ids.new("bool")

            ops.append(
                BooleanOp(
                    op_id=op_id,
                    reads=reads,
                    writes=result_body,
                    kind=BooleanKind(ir_op.boolean_kind.value),
                )
            )

            current_body = result_body
            last_tool_body = None
            body_history.append(result_body)
            continue

        # --------------------------------------------------
        # Export
        # --------------------------------------------------
        if kind == IROpKind.EXPORT:
            if current_body is None:
                raise LoweringError("Export op has no body to export")

            ops.append(
                ExportOp(
                    op_id=op_id,
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

def _lower_topology(ir_op) -> TopoPredicate | None:
    """
    Convert IR topology intent into IRL TopoPredicate.

    Only topology-dependent features (fillet, chamfer) require this.
    Other features must return None.
    """

    # Feature kinds that REQUIRE topology
    kind = ir_op.feature_kind.value
    if kind not in ("fillet", "chamfer"):
        return None

    topo = ir_op.topology
    if topo is None:
        raise LoweringError(
            f"Feature '{kind}' requires topology selection"
        )

    return TopoPredicate(
        target=topo.target,
        rule=topo.rule,
        value=topo.value,
    )
