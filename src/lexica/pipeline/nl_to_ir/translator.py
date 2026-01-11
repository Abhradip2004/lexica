import json
import re
from pathlib import Path
from typing import Any, Dict

from lexica.pipeline.nl_to_ir.ir_normalizer import normalize_ir
from lexica.pipeline.nl_to_ir.schema import IRModel
from lexica.pipeline.nl_to_ir.validate import validate_ir
# from lexica.pipeline.nl_to_ir.errors import TranslationError
from lexica.llm.inference.qwen_local import generate


# -------------------------------------------------
# Config
# -------------------------------------------------

MAX_RETRIES = 3

SYSTEM_PROMPT = (
    Path(__file__)
    .parents[3]
    / "lexica"
    / "llm"
    / "prompts"
    / "system.txt"
).read_text().strip()

class TranslationError(Exception):
    pass


# -------------------------------------------------
# Public API
# -------------------------------------------------

def nl_to_ir(user_input: str, debug: bool = False) -> IRModel:
    """
    Convert natural language input into a validated Lexica IRModel.

    Strategy:
    1. Ask LLM to emit IR
    2. If output already looks like IR → normalize + validate
    3. Otherwise → fallback intent mapper (minimal, safe)
    """

    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = _build_prompt(user_input)
            raw_output = generate(prompt)

            if debug:
                print("[translator] raw output:")
                print(raw_output)

            json_text = _extract_first_json(raw_output)

            if debug:
                print("RAW JSON TEXT:\n", json_text)

            data = json.loads(json_text)

            # -------------------------------------------------
            # Semantic gate
            # -------------------------------------------------

            if not _looks_like_ir(data):
                if debug:
                    print("[translator] non-IR output detected, using fallback")
                return _fallback_intent_to_ir(user_input)

            # -------------------------------------------------
            # IR path
            # -------------------------------------------------

            data = normalize_ir(data)
            ir = IRModel(**data)
            validate_ir(ir)

            return ir

        except Exception as e:
            last_error = str(e)
            if debug:
                print(f"[translator] attempt {attempt} failed: {last_error}")

    raise TranslationError(
        f"Failed to generate valid IR after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# -------------------------------------------------
# Prompting
# -------------------------------------------------

def _build_prompt(user_input: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_input}"
    )


# -------------------------------------------------
# Semantic detection
# -------------------------------------------------

def _looks_like_ir(data: Dict[str, Any]) -> bool:
    """
    Check if JSON already resembles Lexica IR.
    """
    return (
        isinstance(data, dict)
        and "ops" in data
        and isinstance(data["ops"], list)
    )


# -------------------------------------------------
# Fallback intent mapper (TEMPORARY)
# -------------------------------------------------

def _fallback_intent_to_ir(prompt: str) -> IRModel:
    """
    Minimal, safe intent mapper.
    Handles ONLY obvious primitives.
    """

    text = prompt.lower()

    # -------------------------
    # Cube / Box
    # -------------------------

    if "cube" in text:
        m = re.search(r"(\d+)", text)
        side = int(m.group(1)) if m else 10

        return IRModel(
            ops=[
                {
                    "kind": "primitive",
                    "primitive_kind": "box",
                    "params": {
                        "x": side,
                        "y": side,
                        "z": side,
                    },
                }
            ]
        )

    if "box" in text:
        nums = [int(n) for n in re.findall(r"\d+", text)]

        if len(nums) == 3:
            x, y, z = nums
        else:
            x = y = z = nums[0] if nums else 10

        return IRModel(
            ops=[
                {
                    "kind": "primitive",
                    "primitive_kind": "box",
                    "params": {
                        "x": x,
                        "y": y,
                        "z": z,
                    },
                }
            ]
        )

    raise TranslationError("Unable to infer intent")


# -------------------------------------------------
# JSON extraction
# -------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json(text: str) -> str:
    match = _JSON_RE.search(text)
    if not match:
        raise TranslationError("No JSON object found in LLM output")
    return match.group(0)
