"""
Semantic intent dataset for Orbit training
==============================================

Purpose:
- Teach the LLM to understand HUMAN INTENT, not surface grammar
- Map semantic concepts (cube, square block, etc.)
  into CANONICAL Lexica IR
- Reduce failures BEFORE the compiler sees IR

IMPORTANT:
- NEVER emit aliases in IR (no 'cube', no 'square')
- IR must be kernel-safe and schema-valid
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helper: canonical primitive IR builders
# ---------------------------------------------------------------------------

def _box_ir(x: float, y: float, z: float) -> dict:
    return {
        "ops": [
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
    }


# ---------------------------------------------------------------------------
# Intent: Cube (semantic alias of box)
# ---------------------------------------------------------------------------

def cube_intent():
    """
    Human concept: cube
    Canonical IR: box with equal x, y, z
    """

    samples = [
        "Make a cube of side 15",
        "Create a cube with side length 15",
        "cube 15",
        "make cube 15 side",
        "perfect cube size 15",
        "equal sided cube 15",
        "solid cube of 15",
    ]

    return [
        {
            "input": text,
            "ir": _box_ir(15, 15, 15),
        }
        for text in samples
    ]


# ---------------------------------------------------------------------------
# Intent: Box (grammar + phrasing variants)
# ---------------------------------------------------------------------------

def box_intent_variants():
    samples = [
        ("Create a box of size 10 20 30", 10, 20, 30),
        ("box 10 20 30", 10, 20, 30),
        ("make box 10 by 20 by 30", 10, 20, 30),
        ("create rectangular box 10 20 30", 10, 20, 30),
        ("box dimensions 10 20 30", 10, 20, 30),
        ("create a block of 10 20 30", 10, 20, 30),
    ]

    return [
        {
            "input": text,
            "ir": _box_ir(x, y, z),
        }
        for (text, x, y, z) in samples
    ]


# ---------------------------------------------------------------------------
# Intent: Grammar noise / incomplete phrasing
# ---------------------------------------------------------------------------

def grammar_noise_variants():
    """
    These are intentionally ugly.
    Humans speak like this.
    """

    samples = [
        "box make 10 20 30",
        "make 10 20 30 box",
        "create box 10 20 30 please",
        "i want box of 10 20 30",
        "box size is 10 20 30",
        "make a box of size 10, 20, 30",
        "create a box of dimensions 10x20x30",
        "generate a box, the dimensions in x-axis: 10, in y-axis: 20, in z-axis: 30",
        "Make a box of length 10, depth 20, height 30"

    ]

    return [
        {
            "input": text,
            "ir": _box_ir(10, 20, 30),
        }
        for text in samples
    ]


# ---------------------------------------------------------------------------
# Intent: Implicit defaults (SAFE ONLY)
# ---------------------------------------------------------------------------

def implicit_defaults_intent():
    """
    Only defaults that are universally obvious.
    No guessing depth, no guessing dimensions.
    """

    return [
        {
            "input": "Drill a hole of diameter 6",
            "ir": {
                "ops": [
                    {
                        "kind": "feature",
                        "feature_kind": "hole",
                        "params": {
                            "diameter": 6,
                            "through_all": True,
                        },
                    }
                ]
            },
        },
        {
            "input": "Make a through hole of diameter 8",
            "ir": {
                "ops": [
                    {
                        "kind": "feature",
                        "feature_kind": "hole",
                        "params": {
                            "diameter": 8,
                            "through_all": True,
                        },
                    }
                ]
            },
        },
    ]


# ---------------------------------------------------------------------------
# Public API: collect all semantic intent samples
# ---------------------------------------------------------------------------

def all_semantic_intents():
    """
    Collect all semantic intent samples.

    This is what the dataset builder should import.
    """

    samples = []
    samples.extend(cube_intent())
    samples.extend(box_intent_variants())
    samples.extend(grammar_noise_variants())
    samples.extend(implicit_defaults_intent())

    return samples
