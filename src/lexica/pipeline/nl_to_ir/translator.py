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

import requests
import time

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

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 0.5

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranslationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _normalize_ir(model):
    for op in model.ops:
        if hasattr(op, "topology") and op.topology is not None:
            topo = op.topology

            # Only allow edge/face + rule
            allowed_keys = {"target", "rule"}

            # Drop topology if it contains unexpected fields
            if not all(k in allowed_keys for k in topo.__dict__.keys()):
                op.topology = None


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
            # _normalize_ir(ir_model)
            
            for i, op in enumerate(ir_model.ops):
                print(f"[DEBUG] op {i} topology =", getattr(op, "topology", None))
            
            validate_ir(ir_model)
            return ir_model

        except (json.JSONDecodeError, IRValidationError, ValueError) as e:
            last_error = str(e)

            print("\n================ RAW LLM OUTPUT ================\n")
            print(raw_output)
            print("\n================================================\n")

            if debug:
                print(f"[translator] Validation failed (attempt {attempt}): {last_error}")

            time.sleep(RETRY_BACKOFF_SEC)


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
    Call local Qwen via Ollama.
    Returns raw text output.
    """

    prompt = f"{system_prompt}\n\n{user_prompt}".strip()

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 512,
            "stop": [
                "```",
                "Explanation:",
                "<think>",
            ],
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
    except Exception as e:
        raise TranslationError(f"LLM backend error: {e}")

    data = response.json()

    if "response" not in data:
        raise TranslationError("Malformed response from LLM backend")

    return data["response"].strip()



# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    return (
        "You are a deterministic compiler frontend for a CAD system called Lexica.\n\n"
        "Your task is to convert natural language descriptions into a JSON "
        "Intermediate Representation (IR).\n\n"

        "====================\n"
        "ABSOLUTE RULES\n"
        "====================\n"
        "1. Output ONLY valid JSON.\n"
        "2. Do NOT include explanations, comments, markdown, or extra text.\n"
        "3. The output MUST be parseable by json.loads().\n"
        "4. Follow the Lexica IR schema EXACTLY.\n"
        "5. Never invent new fields, operations, or enum values.\n"
        "6. If something is ambiguous, choose the simplest valid interpretation.\n\n"

        "====================\n"
        "IR OPERATION KINDS\n"
        "====================\n"
        "The field \"kind\" MUST be one of the following values ONLY:\n"
        "- \"primitive\"\n"
        "- \"transform\"\n"
        "- \"feature\"\n"
        "- \"boolean\"\n"
        "- \"export\"\n\n"

        "====================\n"
        "IMPORTANT SEMANTIC RULES\n"
        "====================\n"
        "- Primitives create geometry (e.g. box, cylinder).\n"
        "- Features modify existing geometry (e.g. fillet, chamfer, hole).\n"
        "- Transforms move or rotate geometry.\n"
        "- Booleans combine multiple bodies.\n\n"

        "- Topology is ONLY used for selecting edges or faces.\n"
        "- Valid topology targets are ONLY \"edge\" or \"face\".\n"
        "- Positioning concepts like \"center\" are NOT topology.\n"
        "- Positioning such as \"center\" belongs inside params.\n"
        "- Hole features NEVER use topology.\n"
        "- If a feature does not require topology, OMIT the topology field entirely.\n\n"

        "====================\n"
        "TOPOLOGY FORMAT (STRICT)\n"
        "====================\n"
        "If a feature uses topology, the topology field MUST have EXACTLY this shape:\n"
        "{\n"
        "  \"target\": \"edge\" | \"face\",\n"
        "  \"rule\": \"all\" | \"normal\" | \"min\" | \"max\" | "
        "\"parallel\" | \"length_gt\" | \"length_lt\"\n"
        "}\n"
        "Do NOT include extra fields such as index, value, center, position, or axis.\n\n"

        "====================\n"
        "IR SCHEMA EXAMPLES\n"
        "====================\n"

        "Input:\n"
        "\"Create a box of width 10 height 20 length 30\"\n\n"
        "Output:\n"
        "{\n"
        "  \"ops\": [\n"
        "    {\n"
        "      \"kind\": \"primitive\",\n"
        "      \"primitive_kind\": \"box\",\n"
        "      \"params\": {\n"
        "        \"width\": 10,\n"
        "        \"height\": 20,\n"
        "        \"length\": 30\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Input:\n"
        "\"Make a hole in the center with diameter 6\"\n\n"
        "Output:\n"
        "{\n"
        "  \"ops\": [\n"
        "    {\n"
        "      \"kind\": \"feature\",\n"
        "      \"feature_kind\": \"hole\",\n"
        "      \"params\": {\n"
        "        \"diameter\": 6,\n"
        "        \"through_all\": true,\n"
        "        \"center\": [0, 0]\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Input:\n"
        "\"Chamfer all edges by 2\"\n\n"
        "Output:\n"
        "{\n"
        "  \"ops\": [\n"
        "    {\n"
        "      \"kind\": \"feature\",\n"
        "      \"feature_kind\": \"chamfer\",\n"
        "      \"params\": {\n"
        "        \"distance\": 2\n"
        "      },\n"
        "      \"topology\": {\n"
        "        \"target\": \"edge\",\n"
        "        \"rule\": \"all\"\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "====================\n"
        "FINAL OUTPUT FORMAT\n"
        "====================\n"
        "{ \"ops\": [ ... ] }"
    )

def _user_prompt(user_text: str) -> str:
    return (
        "Convert the following description into lexica IR JSON.\n\n"
        f"Description:\n{user_text}"
    )


def _repair_prompt(user_text: str, error: str) -> str:
    return (
        "The previous JSON output was INVALID and rejected by the compiler.\n\n"
        "Compiler error:\n"
        f"{error}\n\n"
        "You MUST output a FULL corrected IR JSON.\n"
        "Follow the Lexica IR schema EXACTLY.\n"
        "Do NOT explain anything.\n"
        "Do NOT include text outside JSON.\n\n"
        "Original description:\n"
        f"{user_text}"
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

