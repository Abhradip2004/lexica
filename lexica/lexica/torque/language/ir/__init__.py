"""
Lexica IR (v0)

This module defines the source language for the Torque compiler.

- The IR captures modeling intent, not execution semantics
- It is deterministic and strictly validated
- It can be authored by humans, tools, or Orbit (LLM frontend)

This IR is lowered into IRL before any kernel execution.
"""

from .schema import (
    IRModel,
    IROp,
    IROpKind,
    PrimitiveKind,
    TransformKind,
    FeatureKind,
    BooleanKind,
    TopologyIntent,
)

from .validate import validate_ir, IRValidationError
