"""
IR Validation (LLM facing)
=========================

Purpose:
- Validate LLM emitted IR against the IR v0 contract
- Reject malformed, incomplete, or hallucinated operations
- Enforce strict schema correctness

This module:
- DOES NOT lower IR
- DOES NOT execute anything
- DOES NOT touch kernel logic
"""

from __future__ import annotations

from typing import Set

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    IROp,
    IROpKind,
    PrimitiveOp,
    TransformOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    PrimitiveKind,
    TransformKind,
    FeatureKind,
    BooleanKind,
    TopologyIntent,
    TopologyTarget,
)


class IRValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_ir(model: IRModel) -> None:
    """
    Validate an IR program.

    Raises:
        IRValidationError on any violation.
    """

    if not model.ops:
        raise IRValidationError("IR model contains no operations")

    _validate_ops_type(model)
    _validate_ops_semantics(model)


# ---------------------------------------------------------------------------
# Validation stages
# ---------------------------------------------------------------------------

def _validate_ops_type(model: IRModel) -> None:
    for idx, op in enumerate(model.ops):
        if not isinstance(op, IROp):
            raise IRValidationError(
                f"IR op at index {idx} is not an IROp"
            )

        if not isinstance(op.kind, IROpKind):
            raise IRValidationError(
                f"IR op at index {idx} has invalid kind: {op.kind}"
            )


def _validate_ops_semantics(model: IRModel) -> None:
    for idx, op in enumerate(model.ops):
        kind = op.kind

        # --------------------------
        # Primitive
        # --------------------------
        if kind == IROpKind.PRIMITIVE:
            if not isinstance(op, PrimitiveOp):
                raise IRValidationError(
                    f"IR op {idx} expected PrimitiveOp"
                )

            if not isinstance(op.primitive_kind, PrimitiveKind):
                raise IRValidationError(
                    f"Primitive op {idx} has invalid primitive_kind"
                )

            _require_params(
                op,
                required={
                    PrimitiveKind.BOX: {"length", "width", "height"},
                    PrimitiveKind.CYLINDER: {"radius", "height"},
                }[op.primitive_kind],
                idx=idx,
            )

        # --------------------------
        # Transform
        # --------------------------
        elif kind == IROpKind.TRANSFORM:
            if not isinstance(op, TransformOp):
                raise IRValidationError(
                    f"IR op {idx} expected TransformOp"
                )

            if not isinstance(op.transform_kind, TransformKind):
                raise IRValidationError(
                    f"Transform op {idx} has invalid transform_kind"
                )

            _require_params(
                op,
                required={
                    TransformKind.TRANSLATE: {"dx", "dy", "dz"},
                    TransformKind.ROTATE: {"axis", "angle_deg"},
                }[op.transform_kind],
                idx=idx,
            )

        # --------------------------
        # Feature
        # --------------------------
        elif kind == IROpKind.FEATURE:
            if not isinstance(op, FeatureOp):
                raise IRValidationError(
                    f"IR op {idx} expected FeatureOp"
                )

            if not isinstance(op.feature_kind, FeatureKind):
                raise IRValidationError(
                    f"Feature op {idx} has invalid feature_kind"
                )

            # Feature params
            _require_params(
                op,
                required={
                    FeatureKind.FILLET: {"radius"},
                    FeatureKind.CHAMFER: {"distance"},
                }[op.feature_kind],
                idx=idx,
            )

            # Topology intent required for topo dependent features
            if op.feature_kind in (FeatureKind.FILLET, FeatureKind.CHAMFER):
                if op.topology is None:
                    raise IRValidationError(
                        f"Feature op {idx} requires topology intent"
                    )
                _validate_topology(op.topology, idx)

        # --------------------------
        # Boolean
        # --------------------------
        elif kind == IROpKind.BOOLEAN:
            if not isinstance(op, BooleanOp):
                raise IRValidationError(
                    f"IR op {idx} expected BooleanOp"
                )

            if not isinstance(op.boolean_kind, BooleanKind):
                raise IRValidationError(
                    f"Boolean op {idx} has invalid boolean_kind"
                )

            if op.operands is not None:
                if not isinstance(op.operands, list) or len(op.operands) < 2:
                    raise IRValidationError(
                        f"Boolean op {idx} operands must be list of >=2 indices"
                    )

        # --------------------------
        # Export
        # --------------------------
        elif kind == IROpKind.EXPORT:
            if not isinstance(op, ExportOp):
                raise IRValidationError(
                    f"IR op {idx} expected ExportOp"
                )

            if not isinstance(op.format, str) or not op.format:
                raise IRValidationError(
                    f"Export op {idx} has invalid format"
                )

            if not isinstance(op.path, str) or not op.path:
                raise IRValidationError(
                    f"Export op {idx} has invalid path"
                )

        else:
            raise IRValidationError(
                f"Unknown IR op kind at index {idx}: {kind}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_params(op: IROp, required: Set[str], idx: int) -> None:
    missing = required - set(op.params.keys())
    if missing:
        raise IRValidationError(
            f"IR op {idx} missing required params: {missing}"
        )


def _validate_topology(topo: TopologyIntent, idx: int) -> None:
    if not isinstance(topo, TopologyIntent):
        raise IRValidationError(
            f"Topology intent in op {idx} is invalid"
        )

    if not isinstance(topo.target, TopologyTarget):
        raise IRValidationError(
            f"Topology intent in op {idx} has invalid target"
        )

    if not isinstance(topo.rule, str) or not topo.rule:
        raise IRValidationError(
            f"Topology intent in op {idx} has invalid rule"
        )

    if topo.value is not None and not isinstance(topo.value, (int, float)):
        raise IRValidationError(
            f"Topology intent in op {idx} has invalid value"
        )
