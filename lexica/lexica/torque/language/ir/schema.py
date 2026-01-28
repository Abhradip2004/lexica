"""
LLM-facing IR (Intermediate Representation)
===========================================

This IR captures *modeling intent*, not execution semantics.

Properties:
- Sequential
- Implicit current body
- Human / LLM friendly
- Deterministically lowerable to IRL
- Kernel-agnostic

This file is a HARD CONTRACT.
If the LLM cannot express something here, it is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# IR operation kinds
# ---------------------------------------------------------------------------

class IROpKind(str, Enum):
    PRIMITIVE = "primitive"
    TRANSFORM = "transform"
    FEATURE   = "feature"
    BOOLEAN   = "boolean"
    EXPORT    = "export"


# ---------------------------------------------------------------------------
# Primitive kinds
# ---------------------------------------------------------------------------

class PrimitiveKind(str, Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CONE = "cone"
    TORUS = "torus"


# ---------------------------------------------------------------------------
# Transform kinds
# ---------------------------------------------------------------------------

class TransformKind(str, Enum):
    TRANSLATE = "translate"
    ROTATE = "rotate"


# ---------------------------------------------------------------------------
# Feature kinds
# ---------------------------------------------------------------------------

class FeatureKind(str, Enum):
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SHELL = "shell"
    HOLE = "hole"


# ---------------------------------------------------------------------------
# Boolean kinds
# ---------------------------------------------------------------------------

class BooleanKind(str, Enum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


# ---------------------------------------------------------------------------
# Topology intent (LLM level)
# ---------------------------------------------------------------------------

class TopologyTarget(str, Enum):
    EDGE = "edge"
    FACE = "face"


@dataclass
class TopologyIntent:
    """
    High level topology intent.

    This is NOT a kernel reference.
    It is an instruction for the compiler to choose topology.
    """
    target: TopologyTarget
    rule: str                # e.g. "all", "convex"
    value: Optional[float] = None


# ---------------------------------------------------------------------------
# IR operation definitions
# ---------------------------------------------------------------------------

@dataclass
class IROp:
    """
    Base IR operation.

    The LLM always emits a flat sequence of these.
    """
    kind: IROpKind = field(init=False)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimitiveOp(IROp):
    primitive_kind: PrimitiveKind = None

    def __post_init__(self):
        self.kind = IROpKind.PRIMITIVE


@dataclass
class TransformOp(IROp):
    transform_kind: TransformKind = None

    def __post_init__(self):
        self.kind = IROpKind.TRANSFORM


@dataclass
class FeatureOp(IROp):
    feature_kind: FeatureKind = None
    topology: Optional[TopologyIntent] = None

    def __post_init__(self):
        self.kind = IROpKind.FEATURE


@dataclass
class BooleanOp(IROp):
    boolean_kind: BooleanKind = None
    operands: Optional[List[int]] = None
    """
    operands are indices into previously created bodies.
    If omitted, defaults to:
        - current body
        - previous body
    """

    def __post_init__(self):
        self.kind = IROpKind.BOOLEAN


@dataclass
class ExportOp(IROp):
    format: str = "step"
    path: str = "output.step"

    def __post_init__(self):
        self.kind = IROpKind.EXPORT


# ---------------------------------------------------------------------------
# IR Model (entire program)
# ---------------------------------------------------------------------------

@dataclass
class IRModel:
    """
    High-level IR program emitted by the LLM.
    """
    ops: List["IROp"]

    def to_dict(self) -> dict:
        """
        Serialize IRModel into plain Python dict.
        Safe for JSON, tests, UI, and logging.
        """
        return asdict(self)
