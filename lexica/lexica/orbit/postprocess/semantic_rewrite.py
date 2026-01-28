"""
lexica.llm.postprocess.semantic_rewrite

This module contains post-processing transforms applied to LLM outputs.
Its purpose is to repair common schema/formatting mistakes and canonicalize
primitive/operator fields before strict kernel validation.

Design principles:
- Kernel remains strict. This module is not part of kernel validation.
- Edits are deterministic and conservative.
- Only safe rewrites are performed (renaming/canonicalization, simple repairs).
- All rewrites are traceable via a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class RewriteEvent:
    path: str
    before: Any
    after: Any
    reason: str


@dataclass
class RewriteReport:
    changed: bool = False
    events: List[RewriteEvent] = field(default_factory=list)

    def add(self, path: str, before: Any, after: Any, reason: str) -> None:
        self.changed = True
        self.events.append(RewriteEvent(path=path, before=before, after=after, reason=reason))


# ---------------------------
# Public API
# ---------------------------

def semantic_rewrite_ir_dict(ir: Dict[str, Any]) -> Tuple[Dict[str, Any], RewriteReport]:
    """
    Rewrite an IR-like dictionary produced by the LLM into a more canonical form.

    The return value is:
        (rewritten_ir_dict, report)

    This function is safe to call even if the model output is incomplete/invalid.
    It does not guarantee validity; strict validation is still required afterwards.
    """
    report = RewriteReport()
    if not isinstance(ir, dict):
        return ir, report

    ir = _deep_copy_dict(ir)

    # Ensure ops exists and is a list-like structure if possible
    if "ops" in ir and isinstance(ir["ops"], tuple):
        before = ir["ops"]
        ir["ops"] = list(ir["ops"])
        report.add("ops", before, ir["ops"], "Converted ops from tuple to list")

    if "ops" not in ir:
        # do not create ops; validation should handle missing ops
        return ir, report

    if not isinstance(ir["ops"], list):
        return ir, report

    # Rewrite op-by-op
    for i, op in enumerate(ir["ops"]):
        if not isinstance(op, dict):
            continue
        _rewrite_single_op(op, base_path=f"ops[{i}]", report=report)

    return ir, report


# ---------------------------
# Rewriting logic
# ---------------------------

# Canonical primitive names expected by kernel
_PRIMITIVE_ALIASES = {
    # canonical: aliases
    "box": {"box", "cuboid", "rectangular_prism", "rect_prism", "cube"},
    "cylinder": {"cylinder", "cyl", "pipe"},
    "sphere": {"sphere", "ball", "globe"},
    "cone": {"cone", "conical"},
    "torus": {"torus", "tori", "donut", "doughnut", "ring"},
}

# Inverse mapping alias->canonical
_PRIMITIVE_ALIAS_TO_CANON: Dict[str, str] = {}
for canon, aliases in _PRIMITIVE_ALIASES.items():
    for a in aliases:
        _PRIMITIVE_ALIAS_TO_CANON[a.lower()] = canon


# Canonical parameter names expected by kernel for each primitive
# (this is where strict kernel naming is enforced)
_CANON_PARAMS = {
    "box": {"x", "y", "z"},
    "cylinder": {"r", "z"},
    "sphere": {"r"},
    "cone": {"r1", "r2", "z"},
    "torus": {"R", "r"},
}

# Allowed key rewrites: primitive param canonicalization
_PARAM_KEY_REWRITES = {
    # general
    "radius": "r",
    "rad": "r",
    "height": "z",
    "h": "z",

    # cone common names
    "base_radius": "r1",
    "bottom_radius": "r1",
    "r_base": "r1",
    "top_radius": "r2",
    "upper_radius": "r2",
    "r_top": "r2",

    # torus common names
    "major_radius": "R",
    "R_major": "R",
    "minor_radius": "r",
    "tube_radius": "r",
}

# Sometimes model returns kind names in different keys
_OP_KIND_KEY_ALIASES = ["kind", "op_kind", "type", "opType"]


def _rewrite_single_op(op: Dict[str, Any], base_path: str, report: RewriteReport) -> None:
    """
    Rewrite one operation dict in-place.
    """
    # Normalize kind key
    if "kind" not in op:
        for k in _OP_KIND_KEY_ALIASES:
            if k in op:
                before = op[k]
                op["kind"] = op.pop(k)
                report.add(f"{base_path}.{k}", before, op["kind"], f"Moved '{k}' to 'kind'")
                break

    # Canonicalize primitive ops
    kind = op.get("kind")
    if isinstance(kind, str) and kind.lower() == "primitive":
        _rewrite_primitive_op(op, base_path, report)


def _rewrite_primitive_op(op: Dict[str, Any], base_path: str, report: RewriteReport) -> None:
    """
    Primitive op canonicalization:
    - normalize primitive_kind field
    - normalize params object and param keys
    """
    # Normalize primitive kind key
    # Your IR uses `primitive_kind`. LLM may output `primitive` or `primitiveKind`.
    if "primitive_kind" not in op:
        for alt in ("primitive", "primitiveKind", "primitive_type", "primitiveType"):
            if alt in op:
                before = op[alt]
                op["primitive_kind"] = op.pop(alt)
                report.add(f"{base_path}.{alt}", before, op["primitive_kind"], f"Moved '{alt}' to 'primitive_kind'")
                break

    prim = op.get("primitive_kind")

    # primitive_kind should be string-like at this stage
    if isinstance(prim, str):
        canon = _PRIMITIVE_ALIAS_TO_CANON.get(prim.lower())
        if canon and canon != prim:
            before = prim
            op["primitive_kind"] = canon.upper() if prim.isupper() else canon
            report.add(
                f"{base_path}.primitive_kind",
                before,
                op["primitive_kind"],
                "Canonicalized primitive kind alias",
            )

        # bring to lowercase canonical names for kernel (schema expects enums anyway)
        prim2 = op["primitive_kind"]
        if isinstance(prim2, str):
            canon2 = _PRIMITIVE_ALIAS_TO_CANON.get(prim2.lower(), prim2.lower())
            if canon2 != prim2:
                before = prim2
                op["primitive_kind"] = canon2
                report.add(
                    f"{base_path}.primitive_kind",
                    before,
                    canon2,
                    "Normalized primitive_kind casing",
                )

    # Normalize params key
    if "params" not in op:
        for alt in ("parameters", "args", "arguments", "param"):
            if alt in op and isinstance(op[alt], dict):
                before = op[alt]
                op["params"] = op.pop(alt)
                report.add(f"{base_path}.{alt}", before, op["params"], f"Moved '{alt}' to 'params'")
                break

    if "params" not in op or not isinstance(op["params"], dict):
        return

    params = op["params"]

    # Rewrite param keys using dictionary
    for old_key in list(params.keys()):
        if not isinstance(old_key, str):
            continue
        lk = old_key.lower()

        if lk in _PARAM_KEY_REWRITES:
            new_key = _PARAM_KEY_REWRITES[lk]
            if new_key not in params:
                before = params[old_key]
                params[new_key] = params.pop(old_key)
                report.add(
                    f"{base_path}.params.{old_key}",
                    before,
                    params[new_key],
                    f"Renamed param '{old_key}' to '{new_key}'",
                )

    # Special rewrite rules (torus expects R, not r1)
    prim_kind = op.get("primitive_kind")
    if isinstance(prim_kind, str) and prim_kind.lower() == "torus":
        # Handle cases like { "r1": ..., "r2": ... } produced for torus
        if "R" not in params and "r1" in params:
            before = params["r1"]
            params["R"] = params.pop("r1")
            report.add(
                f"{base_path}.params.r1",
                before,
                params["R"],
                "Mapped torus major radius from r1 to R",
            )
        if "r" not in params and "r2" in params:
            before = params["r2"]
            params["r"] = params.pop("r2")
            report.add(
                f"{base_path}.params.r2",
                before,
                params["r"],
                "Mapped torus minor radius from r2 to r",
            )

    # Normalize numeric values (string numbers -> float/int)
    for k, v in list(params.items()):
        if isinstance(v, str):
            vv = v.strip()
            if _looks_like_number(vv):
                before = v
                params[k] = _parse_number(vv)
                report.add(
                    f"{base_path}.params.{k}",
                    before,
                    params[k],
                    "Parsed numeric string into number",
                )


# ---------------------------
# Helpers
# ---------------------------

def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal deep copy for nested dict/list structure.
    Avoids importing deepcopy for performance and to keep rewrite deterministic.
    """
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = [_deep_copy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _looks_like_number(s: str) -> bool:
    # Supports "10", "10.5", "-3.2"
    if not s:
        return False
    if s.count(".") > 1:
        return False
    if s[0] == "-":
        s = s[1:]
    return s.replace(".", "", 1).isdigit()


def _parse_number(s: str) -> int | float:
    if "." in s:
        return float(s)
    return int(s)
