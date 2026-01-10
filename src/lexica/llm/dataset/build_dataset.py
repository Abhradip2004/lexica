"""
Lexica Orbit Dataset Builder
=========================

Purpose:
- Build high-quality training data for NL --> IR
- Enforce IR v1 correctness at dataset creation time
- Generate BOTH positive and negative samples
- Guarantee: every positive sample passes validate_ir()

This file is the ONLY place where training data is constructed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    PrimitiveKind,
    FeatureKind,
    BooleanKind,
    TopologyIntent,
    TopologyTarget,
)
from lexica.pipeline.nl_to_ir.validate import validate_ir, IRValidationError
from lexica.llm.dataset.intent_semantic import all_semantic_intents


# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).parent
TRAIN_OUT = BASE_DIR / "train.jsonl"
EVAL_OUT = BASE_DIR / "eval.jsonl"

SYSTEM_PROMPT = (
    Path(__file__)
    .parents[2]
    /"prompts"
    / "system.txt"
).read_text().strip()


# -------------------------------------------------
# Dataset helpers
# -------------------------------------------------

def _msg(system: str, user: str, assistant: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _serialize_ir(ir: IRModel) -> str:
    """
    Serialize IRModel into deterministic JSON.
    """
    def serialize_op(op):
        d = {"kind": op.kind.value}

        if isinstance(op, PrimitiveOp):
            d["primitive_kind"] = op.primitive_kind.value

        if isinstance(op, FeatureOp):
            d["feature_kind"] = op.feature_kind.value
            if op.topology:
                d["topology"] = {
                    "target": op.topology.target.value,
                    "rule": op.topology.rule,
                }

        if isinstance(op, BooleanOp):
            d["boolean_kind"] = op.boolean_kind.value
            if op.operands:
                d["operands"] = op.operands

        d["params"] = op.params
        return d

    return json.dumps(
        {"ops": [serialize_op(op) for op in ir.ops]},
        separators=(",", ":"),
    )

def _semantic_intent_samples() -> List[Dict]:
    """
    Convert semantic intent definitions into validated training samples.
    """

    samples = []

    for item in all_semantic_intents():
        user_text = item["input"]
        ir_dict = item["ir"]

        # Convert dict -> IRModel to validate
        ir_model = IRModel(
            ops=[
                PrimitiveOp(
                    primitive_kind=PrimitiveKind(op["primitive_kind"]),
                    params=op["params"],
                )
                if op["kind"] == "primitive"
                else FeatureOp(
                    feature_kind=FeatureKind(op["feature_kind"]),
                    params=op["params"],
                )
                for op in ir_dict["ops"]
            ]
        )

        # HARD GUARANTEE: semantic samples must be valid IR
        validate_ir(ir_model)

        samples.append(
            _msg(
                SYSTEM_PROMPT,
                user_text,
                _serialize_ir(ir_model),
            )
        )

    return samples


# -------------------------------------------------
# Positive samples
# -------------------------------------------------

def positive_samples() -> List[Dict]:
    samples = []

    # --------------------------
    # Canonical samples
    # --------------------------
    ir = IRModel(
        ops=[
            PrimitiveOp(
                primitive_kind=PrimitiveKind.BOX,
                params={"x": 30, "y": 30, "z": 30},
            )
        ]
    )
    validate_ir(ir)
    samples.append(
        _msg(
            SYSTEM_PROMPT,
            "Create a 30 by 30 by 30 box",
            _serialize_ir(ir),
        )
    )

    ir = IRModel(
        ops=[
            PrimitiveOp(
                primitive_kind=PrimitiveKind.BOX,
                params={"x": 20, "y": 20, "z": 20},
            ),
            FeatureOp(
                feature_kind=FeatureKind.FILLET,
                params={"radius": 2},
                topology=TopologyIntent(
                    target=TopologyTarget.EDGE,
                    rule="all",
                ),
            ),
        ]
    )
    validate_ir(ir)
    samples.append(
        _msg(
            SYSTEM_PROMPT,
            "Create a 20mm cube and fillet all edges by 2mm",
            _serialize_ir(ir),
        )
    )

    samples.extend(_semantic_intent_samples())

    return samples



# -------------------------------------------------
# Negative samples (compiler rejection)
# -------------------------------------------------

def negative_samples() -> List[Dict]:
    samples = []

    # --------------------------
    # Fillet without solid
    # --------------------------
    samples.append(
        _msg(
            SYSTEM_PROMPT,
            "Fillet all edges by 2mm",
            json.dumps({"error": "Feature requires an existing solid"}),
        )
    )

    # --------------------------
    # Unknown operation
    # --------------------------
    samples.append(
        _msg(
            SYSTEM_PROMPT,
            "Smooth the box",
            json.dumps({"error": "Unknown operation: smooth"}),
        )
    )

    # --------------------------
    # Invalid primitive params
    # --------------------------
    samples.append(
        _msg(
            SYSTEM_PROMPT,
            "Create a box of width 10 height 10 depth 10",
            json.dumps(
                {
                    "error": "Primitive BOX uses invalid params. Use x, y, z."
                }
            ),
        )
    )

    return samples


# -------------------------------------------------
# Build dataset
# -------------------------------------------------

def build():
    all_samples = []
    all_samples.extend(positive_samples())
    all_samples.extend(negative_samples())

    random.shuffle(all_samples)

    split = int(len(all_samples) * 0.9)
    train = all_samples[:split]
    eval_ = all_samples[split:]

    with TRAIN_OUT.open("w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")

    with EVAL_OUT.open("w") as f:
        for s in eval_:
            f.write(json.dumps(s) + "\n")

    print(f"[dataset] train samples: {len(train)}")
    print(f"[dataset] eval samples:  {len(eval_)}")


if __name__ == "__main__":
    build()
