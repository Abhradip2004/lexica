"""
IR Validation (Language-level)
=============================

Purpose:
- Validate IR programs against the Lexica IR v0 specification
- Reject malformed or incomplete IR
- Enforce strict language correctness

This module:
- DOES NOT lower IR
- DOES NOT execute anything
- DOES NOT interact with the kernel
"""


from __future__ import annotations

from typing import Set

from lexica.torque.language.ir.schema import (
    IRModel,
    IROp,
    IROpKind,
    PrimitiveKind,
    FeatureKind,
    TransformKind,
    BooleanKind,
    ExportOp,
)


class IRValidationError(Exception):
    pass


# Mapping of supported primitive types to the set of parameters that must be
# present for each primitive operation. This acts as a strict kernel-facing
# contract: if a required parameter is missing, the IR must be rejected.
#
# Note: This validation layer is intentionally strict and does not attempt to
# infer missing values or rename keys. Any lenient rewriting should be done
# in the LLM/NL preprocessing layer.
PRIMITIVE_REQUIRED_PARAMS: dict[PrimitiveKind, Set[str]] = {
    PrimitiveKind.BOX: {"x", "y", "z"},
    PrimitiveKind.CYLINDER: {"r", "z"},

    # New primitives
    PrimitiveKind.SPHERE: {"r"},
    PrimitiveKind.CONE: {"r1", "r2", "z"},
    PrimitiveKind.TORUS: {"R", "r"},
}


def validate_ir(model: IRModel) -> None:
    """
    Validate the IR model strictly.

    This function enforces the kernel contract. It verifies:
    - IR structure is valid (ops list exists, contains correct op types)
    - each op has a valid kind (IROpKind)
    - for each supported op category (primitive/feature/transform/boolean/export),
      required fields exist and are of valid types
    - primitive params are present, numeric, and satisfy geometric constraints

    This validator intentionally avoids any semantic rewrite or tolerant parsing.
    The role of this function is to ensure execution safety and determinism.
    """

    if model is None:
        raise IRValidationError("IR model is None")

    # Base-level structural checks
    _validate_ops_type(model)
    _validate_ops_kind(model)

    # Category-specific checks
    _validate_primitive_ops(model)
    _validate_feature_ops(model)
    _validate_transform_ops(model)
    _validate_boolean_ops(model)
    _validate_export_ops(model)


def _validate_ops_type(model: IRModel) -> None:
    """
    Ensure 'ops' exists and contains only IROp objects.

    This prevents downstream logic from failing with attribute errors and ensures
    every operation in the pipeline conforms to the expected schema layer.
    """

    if not hasattr(model, "ops") or model.ops is None:
        raise IRValidationError("IR model missing 'ops' list")

    if not isinstance(model.ops, list):
        raise IRValidationError("IR model 'ops' must be a list")

    for idx, op in enumerate(model.ops):
        if not isinstance(op, IROp):
            raise IRValidationError(f"IR op at index {idx} is not an IROp")


def _validate_ops_kind(model: IRModel) -> None:
    """
    Ensure every op has a kind and the kind is an instance of IROpKind.
    """

    for idx, op in enumerate(model.ops):
        if not hasattr(op, "kind"):
            raise IRValidationError(f"IR op at index {idx} missing 'kind'")

        if not isinstance(op.kind, IROpKind):
            raise IRValidationError(
                f"IR op at index {idx} has invalid kind: {op.kind}"
            )


def _require_params(op: IROp, required: Set[str], idx: int) -> None:
    """
    Ensure op has a params dictionary with all required keys.
    """

    if not hasattr(op, "params") or op.params is None:
        raise IRValidationError(f"IR op {idx} missing 'params'")

    if not isinstance(op.params, dict):
        raise IRValidationError(f"IR op {idx} params must be a dict")

    missing = required - set(op.params.keys())
    if missing:
        raise IRValidationError(f"IR op {idx} missing required params: {missing}")


def _require_numeric_params(op: IROp, keys: Set[str], idx: int) -> None:
    """
    Ensure that required params are numeric.

    The kernel relies on numeric types for geometry construction. Values must be
    int or float. Bool is rejected explicitly because Python treats bool as a
    subclass of int.
    """

    for k in keys:
        v = op.params[k]

        if isinstance(v, bool):
            raise IRValidationError(f"IR op {idx} param '{k}' cannot be bool")

        if not isinstance(v, (int, float)):
            raise IRValidationError(
                f"IR op {idx} param '{k}' must be numeric (int/float), got {type(v).__name__}"
            )


def _validate_primitive_ops(model: IRModel) -> None:
    """
    Validate primitive operations.

    Primitive ops are the foundation of geometry generation, therefore they must
    be validated rigorously:
    - supported primitive kind
    - required parameter presence
    - numeric parameter type
    - geometric constraints (positive dimensions, etc.)
    """

    for idx, op in enumerate(model.ops):
        if op.kind != IROpKind.PRIMITIVE:
            continue

        if not hasattr(op, "primitive_kind"):
            raise IRValidationError(f"Primitive op at index {idx} missing primitive_kind")

        if not isinstance(op.primitive_kind, PrimitiveKind):
            raise IRValidationError(f"Primitive op at index {idx} has invalid primitive_kind")

        prim_kind = op.primitive_kind

        if prim_kind not in PRIMITIVE_REQUIRED_PARAMS:
            raise IRValidationError(
                f"Primitive op at index {idx} unsupported primitive: {prim_kind}"
            )

        required = PRIMITIVE_REQUIRED_PARAMS[prim_kind]

        _require_params(op, required, idx)
        _require_numeric_params(op, required, idx)

        # Primitive-specific geometric constraints
        if prim_kind == PrimitiveKind.BOX:
            # A box must have strictly positive dimensions.
            if op.params["x"] <= 0 or op.params["y"] <= 0 or op.params["z"] <= 0:
                raise IRValidationError(f"Box op {idx}: x,y,z must be > 0")

        elif prim_kind == PrimitiveKind.CYLINDER:
            # Cylinder requires positive radius and height.
            if op.params["r"] <= 0:
                raise IRValidationError(f"Cylinder op {idx}: radius r must be > 0")
            if op.params["z"] <= 0:
                raise IRValidationError(f"Cylinder op {idx}: height z must be > 0")

        elif prim_kind == PrimitiveKind.SPHERE:
            # Sphere radius must be strictly positive.
            if op.params["r"] <= 0:
                raise IRValidationError(f"Sphere op {idx}: radius r must be > 0")

        elif prim_kind == PrimitiveKind.CONE:
            # Cone is defined by base radius (r1), top radius (r2), and height (z).
            # Allow r2 == 0 for a proper cone. Disallow both radii == 0.
            r1 = op.params["r1"]
            r2 = op.params["r2"]
            z = op.params["z"]

            if z <= 0:
                raise IRValidationError(f"Cone op {idx}: height z must be > 0")
            if r1 < 0 or r2 < 0:
                raise IRValidationError(f"Cone op {idx}: r1/r2 must be >= 0")
            if r1 == 0 and r2 == 0:
                raise IRValidationError(f"Cone op {idx}: invalid cone, r1 and r2 both 0")

        elif prim_kind == PrimitiveKind.TORUS:
            # Torus is parameterized by:
            # R = major radius (center to tube center)
            # r = minor radius (tube radius)
            #
            # Valid torus requires r < R.
            R = op.params["R"]
            r = op.params["r"]

            if R <= 0 or r <= 0:
                raise IRValidationError(f"Torus op {idx}: R and r must be > 0")
            if r >= R:
                raise IRValidationError(f"Torus op {idx}: must satisfy r < R")


def _validate_feature_ops(model: IRModel) -> None:
    """
    Validate feature operations.

    This validator checks only that feature_kind exists and is valid. Parameter
    validation for features should be implemented in feature-specific validation
    when feature ops are expanded.
    """

    for idx, op in enumerate(model.ops):
        if op.kind != IROpKind.FEATURE:
            continue

        if not hasattr(op, "feature_kind"):
            raise IRValidationError(f"Feature op {idx} missing feature_kind")

        if not isinstance(op.feature_kind, FeatureKind):
            raise IRValidationError(f"Feature op {idx} invalid feature_kind")


def _validate_transform_ops(model: IRModel) -> None:
    """
    Validate transform operations.

    Only kind presence/type is validated here. Transform parameter validation is
    typically strict but depends on individual transform types.
    """

    for idx, op in enumerate(model.ops):
        if op.kind != IROpKind.TRANSFORM:
            continue

        if not hasattr(op, "transform_kind"):
            raise IRValidationError(f"Transform op {idx} missing transform_kind")

        if not isinstance(op.transform_kind, TransformKind):
            raise IRValidationError(f"Transform op {idx} invalid transform_kind")


def _validate_boolean_ops(model: IRModel) -> None:
    """
    Validate boolean operations.

    Boolean operations require valid boolean_kind. More detailed semantic
    validation (operand availability, operand compatibility) can be added once
    boolean op semantics are expanded.
    """

    for idx, op in enumerate(model.ops):
        if op.kind != IROpKind.BOOLEAN:
            continue

        if not hasattr(op, "boolean_kind"):
            raise IRValidationError(f"Boolean op {idx} missing boolean_kind")

        if not isinstance(op.boolean_kind, BooleanKind):
            raise IRValidationError(f"Boolean op {idx} invalid boolean_kind")


def _validate_export_ops(model: IRModel) -> None:
    """
    Validate export operations.

    ExportOp currently uses `format` (string), not an enum `export_kind`.
    """
    for idx, op in enumerate(model.ops):
        if op.kind != IROpKind.EXPORT:
            continue

        if not hasattr(op, "format"):
            raise IRValidationError(f"Export op {idx} missing format")

        if not isinstance(op.format, str):
            raise IRValidationError(f"Export op {idx} format must be a string")

        if op.format.strip().lower() not in {"step"}:
            raise IRValidationError(
                f"Export op {idx} unsupported format: {op.format}. Supported: step"
            )

