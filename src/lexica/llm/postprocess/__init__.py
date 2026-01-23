"""
lexica.llm.postprocess

Post-processing utilities applied to Orbit outputs.

This package contains deterministic repair and canonicalization passes used to
improve the reliability of IR produced by language models before strict kernel
validation and execution.

These modules must not weaken kernel constraints. The kernel remains strict;
postprocessing exists only to:
- fix common key/alias mistakes (e.g., "radius" -> "r")
- canonicalize primitive/operator names (e.g., "donut" -> "torus")
- parse numeric strings into numeric types (e.g., "10" -> 10)
- provide rewrite reports for debugging and traceability

Public API:
- semantic_rewrite_ir_dict: rewrite IR dictionaries and return a rewrite report
"""

from .semantic_rewrite import (
    semantic_rewrite_ir_dict,
    RewriteEvent,
    RewriteReport,
)

__all__ = [
    "semantic_rewrite_ir_dict",
    "RewriteEvent",
    "RewriteReport",
]
