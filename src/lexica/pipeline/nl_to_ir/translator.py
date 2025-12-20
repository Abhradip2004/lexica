"""
NL --> IR Translator (LLM-backed)
==============================

Purpose:
- Convert natural language into IR v0 using an LLM
- Enforce strict JSON-only output
- Validate IR schema
- Retry on failure with bounded attempts

This module:
- DOES NOT lower IR
- DOES NOT execute anything
- DOES NOT touch kernel logic
"""

from __future__ import annotations

import json
from typing import Optional

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
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
from lexica.pipeline.nl_to_ir.validate import validate_ir, IRValidationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranslationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def nl_to_ir(prompt: str, *, debug: bool = False) -> IRModel:
    """
    Convert natural language prompt to IRModel.

    Raises:
        TranslationError if valid IR cannot be produced.
    """

    system_prompt = _system_prompt()
    user_prompt = _user_prompt(prompt)

    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt if last_error is None else _repair_prompt(prompt, last_error),
        )

        if debug:
            print(f"[translator] LLM raw output (attempt {attempt}):")
            print(raw_output)

        try:
            ir_dict = _parse_json(raw_output)
            ir_model = _dict_to_ir(ir_dict)
            validate_ir(ir_model)
            return ir_model

        except (json.JSONDecodeError, IRValidationError, ValueError) as e:
            last_error = str(e)
            if debug:
                print(f"[translator] Validation failed: {last_error}")

    raise TranslationError(
        f"Failed to generate valid IR after {MAX_RETRIES} attempts.\n"
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# LLM interaction (backend-agnostic)
# ---------------------------------------------------------------------------

# def call_llm(*, system_prompt: str, user_prompt: str) -> str:
#     """
#     Call the LLM and return raw text output.
#     """

#     raise NotImplementedError(
#         "LLM backend not configured. Implement call_llm()."
#     )

def call_llm(*, system_prompt: str, user_prompt: str) -> str:
    """
    Mock LLM backend for deterministic testing.
    """

    text = user_prompt.lower()

    # Test case: box → translate → rotate → fillet → export
    if "box" in text and "fillet" in text:
        return """
        {
        "ops": [
            {
            "kind": "primitive",
            "primitive_kind": "box",
            "params": { "length": 40, "width": 30, "height": 20 }
            },
            {
            "kind": "transform",
            "transform_kind": "translate",
            "params": { "dx": 25, "dy": 0, "dz": 0 }
            },
            {
            "kind": "transform",
            "transform_kind": "rotate",
            "params": { "axis": "z", "angle_deg": 45 }
            },
            {
            "kind": "feature",
            "feature_kind": "fillet",
            "params": { "radius": 2.5 },
            "topology": { "target": "edge", "rule": "all" }
            },
            {
            "kind": "export",
            "format": "step",
            "path": "mock_llm_output.step"
            }
        ]
        }
        """

    raise RuntimeError("Mock LLM received unknown prompt")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    return (
        "You are a CAD intent compiler for a system called lexica.\n\n"
        "Your job is to convert natural language descriptions of 3D models\n"
        "into a JSON Intermediate Representation (IR).\n\n"
        "CRITICAL RULES:\n"
        "1. Output ONLY valid JSON.\n"
        "2. Follow the lexica IR schema exactly.\n"
        "3. Do NOT output explanations, comments, or markdown.\n"
        "4. Do NOT output CAD or kernel code.\n"
        "5. Do NOT invent operations.\n"
        "6. Use explicit numeric parameters.\n"
        "7. Assume operations apply to the most recent body.\n"
        "8. If ambiguous, choose the simplest valid interpretation.\n\n"
        "Output format:\n"
        "{ \"ops\": [ ... ] }"
    )


def _user_prompt(user_text: str) -> str:
    return (
        "Convert the following description into lexica IR JSON.\n\n"
        f"Description:\n{user_text}"
    )


def _repair_prompt(user_text: str, error: str) -> str:
    return (
        "The previous output was invalid.\n\n"
        f"Validation error:\n{error}\n\n"
        "Regenerate the full IR JSON correctly.\n\n"
        f"Original description:\n{user_text}"
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """
    Parse JSON strictly.

    Raises:
        JSONDecodeError if invalid.
    """
    return json.loads(text)


def _dict_to_ir(data: dict) -> IRModel:
    """
    Convert raw JSON dict into IRModel with schema dataclasses.

    Raises:
        ValueError on any schema violation.
    """

    if "ops" not in data or not isinstance(data["ops"], list):
        raise ValueError("IR JSON must contain 'ops' list")

    ops = []

    for idx, raw_op in enumerate(data["ops"]):
        if not isinstance(raw_op, dict):
            raise ValueError(f"IR op {idx} must be an object")

        kind = raw_op.get("kind")
        if kind is None:
            raise ValueError(f"IR op {idx} missing 'kind' field")

        try:
            kind_enum = IROpKind(kind)
        except ValueError:
            raise ValueError(f"IR op {idx} has invalid kind '{kind}'")

        # --------------------------
        # Primitive
        # --------------------------
        if kind_enum == IROpKind.PRIMITIVE:
            pk = raw_op.get("primitive_kind")
            if pk is None:
                raise ValueError(f"Primitive op {idx} missing primitive_kind")

            try:
                primitive_kind = PrimitiveKind(pk)
            except ValueError:
                raise ValueError(f"Primitive op {idx} has invalid primitive_kind '{pk}'")

            op = PrimitiveOp(
                primitive_kind=primitive_kind,
                params=_require_dict(raw_op, "params", idx),
            )

        # --------------------------
        # Transform
        # --------------------------
        elif kind_enum == IROpKind.TRANSFORM:
            tk = raw_op.get("transform_kind")
            if tk is None:
                raise ValueError(f"Transform op {idx} missing transform_kind")

            try:
                transform_kind = TransformKind(tk)
            except ValueError:
                raise ValueError(f"Transform op {idx} has invalid transform_kind '{tk}'")

            op = TransformOp(
                transform_kind=transform_kind,
                params=_require_dict(raw_op, "params", idx),
            )

        # --------------------------
        # Feature
        # --------------------------
        elif kind_enum == IROpKind.FEATURE:
            fk = raw_op.get("feature_kind")
            if fk is None:
                raise ValueError(f"Feature op {idx} missing feature_kind")

            try:
                feature_kind = FeatureKind(fk)
            except ValueError:
                raise ValueError(f"Feature op {idx} has invalid feature_kind '{fk}'")

            topo = None
            if "topology" in raw_op:
                topo_raw = raw_op["topology"]
                topo = _parse_topology(topo_raw, idx)

            op = FeatureOp(
                feature_kind=feature_kind,
                topology=topo,
                params=_require_dict(raw_op, "params", idx),
            )

        # --------------------------
        # Boolean
        # --------------------------
        elif kind_enum == IROpKind.BOOLEAN:
            bk = raw_op.get("boolean_kind")
            if bk is None:
                raise ValueError(f"Boolean op {idx} missing boolean_kind")

            try:
                boolean_kind = BooleanKind(bk)
            except ValueError:
                raise ValueError(f"Boolean op {idx} has invalid boolean_kind '{bk}'")

            operands = raw_op.get("operands")

            op = BooleanOp(
                boolean_kind=boolean_kind,
                operands=operands,
                params={},
            )

        # --------------------------
        # Export
        # --------------------------
        elif kind_enum == IROpKind.EXPORT:
            fmt = raw_op.get("format")
            path = raw_op.get("path")

            if not fmt or not path:
                raise ValueError(f"Export op {idx} requires format and path")

            op = ExportOp(
                format=fmt,
                path=path,
                params={},
            )

        else:
            raise ValueError(f"Unhandled IR op kind '{kind_enum}'")

        ops.append(op)

    return IRModel(ops=ops)

def _require_dict(raw_op: dict, field: str, idx: int) -> dict:
    value = raw_op.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"IR op {idx} requires '{field}' object")
    return value


def _parse_topology(raw: dict, idx: int) -> TopologyIntent:
    if not isinstance(raw, dict):
        raise ValueError(f"Topology in op {idx} must be an object")

    target = raw.get("target")
    rule = raw.get("rule")
    value = raw.get("value")

    if target is None or rule is None:
        raise ValueError(f"Topology in op {idx} requires target and rule")

    try:
        target_enum = TopologyTarget(target)
    except ValueError:
        raise ValueError(f"Topology in op {idx} has invalid target '{target}'")

    return TopologyIntent(
        target=target_enum,
        rule=rule,
        value=value,
    )

