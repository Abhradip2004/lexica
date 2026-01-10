from __future__ import annotations

import json
from typing import Optional

from lexica.llm.inference.qwen_local import generate
from lexica.pipeline.nl_to_ir.schema import IRModel
from lexica.pipeline.nl_to_ir.validate import validate_ir
from lexica.pipeline.nl_to_ir.normalize import normalize_ir
from pathlib import Path



# -------------------------------------------------
# Configuration
# -------------------------------------------------

MAX_RETRIES = 3

SYSTEM_PROMPT = (
    Path(__file__)
    .parents[3]  
    / "lexica"
    / "llm"
    /"prompts"
    / "system.txt"
).read_text().strip()

# -------------------------------------------------
# Errors
# -------------------------------------------------

class TranslationError(RuntimeError):
    pass


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return text[start:end + 1]

def _extract_first_json(text: str) -> str:
    """
    Extract the FIRST complete JSON object from text.
    Ignores anything after it.
    """
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]

    raise ValueError("No complete JSON object found")

def _normalize_ir_dict(data: dict) -> dict:
    """
    Normalize LLM-produced IR variants into canonical Lexica IR.
    This is INTENT NORMALIZATION, not validation.
    """

    # operation -> ops
    if "operation" in data and "ops" not in data:
        data["ops"] = [data["operation"]]
        del data["operation"]

    # operations -> ops
    if "operations" in data and "ops" not in data:
        data["ops"] = data["operations"]
        del data["operations"]

    # single op object -> list
    if "ops" in data and isinstance(data["ops"], dict):
        data["ops"] = [data["ops"]]

    return data


# -------------------------------------------------
# Translator
# -------------------------------------------------

def _is_error_object(data: dict) -> bool:
    return (
        isinstance(data, dict)
        and "error" in data
        and len(data) == 1
    )


def nl_to_ir(user_input: str, debug: bool = False) -> IRModel:
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{user_input}"
            )

            raw_output = generate(prompt)

            json_text = _extract_first_json(raw_output)
            data = json.loads(json_text)
            
            if _is_error_object(data):
                raise TranslationError(f"LLM returned error: {data['error']}")

            data = _normalize_ir_dict(data)
            ir = IRModel(**data)
            validate_ir(ir)
            ir = normalize_ir(ir)

            return ir

        except Exception as e:
            last_error = e
            if debug:
                print(f"[translator] attempt {attempt} failed: {e}")
            continue

    raise TranslationError(
        f"Failed to generate valid IR after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
