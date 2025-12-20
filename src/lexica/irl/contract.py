"""
IRL (Intermediate Representation) Contract
====================================================

This file defines the *semantic boundary* between the compiler frontend
(NL --> IR --> IRL) and the CAD executor.

Key guarantees:
- No implicit CAD state
- No execution logic
- Explicit body identity
- Feature ops are first-class citizens
- Deterministic, kernel-grade semantics

If an operation cannot be expressed here, it is NOT allowed in lexica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any


# ---------------------------------------------------------------------------
# Core identity types
# ---------------------------------------------------------------------------

BodyID = str
OpID = str


# ---------------------------------------------------------------------------
# Operation categories (semantic)
# ---------------------------------------------------------------------------

class IRLOpCategory(str, Enum):
    PRIMITIVE = "primitive"   # creates new topology
    FEATURE   = "feature"     # modifies existing topology
    BOOLEAN   = "boolean"     # multi-body topology ops
    EXPORT    = "export"      # serialization only


# ---------------------------------------------------------------------------
# Topological selection 
# ---------------------------------------------------------------------------

class TopoTarget(str, Enum):
    EDGE = "edge"
    FACE = "face"
    SOLID = "solid"


@dataclass(frozen=True)
class TopoPredicate:
    """
    Topology selection.

    This is intentionally limited:
    - No stable naming
    - No kernel object references
    - Deterministic and stateless

    Examples:
    - all edges
    - edges with length > X
    - convex edges
    """
    target: TopoTarget
    rule: Literal[
        "all",
        "convex",
        "concave",
        "by_length_gt",
        "by_length_lt",
    ]
    value: Optional[float] = None


# ---------------------------------------------------------------------------
# Base IRL operation (ALL ops must conform to this)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRLOp:
    """
    Canonical IRL operation.

    Rules (NON-NEGOTIABLE):
    - reads MUST be explicit
    - writes MUST be explicit
    - no implicit mutation
    - exactly one output body for non-export ops
    """

    op_id: OpID
    category: IRLOpCategory

    # Body access contract
    reads: List[BodyID]
    writes: Optional[BodyID]

    # Operation-specific parameters (kernel validated later)
    params: Dict[str, Any] = field(default_factory=dict)

    # Optional topological targeting (used by feature ops)
    topo: Optional[TopoPredicate] = None


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveOp(IRLOp):
    """
    Geometry creation.

    Semantics:
    - reads MUST be empty
    - writes MUST create a new body
    """

    category: IRLOpCategory = field(
        default=IRLOpCategory.PRIMITIVE, init=False
    )


# ---------------------------------------------------------------------------
# Feature operations (fillet, chamfer, shell, draft, etc.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureOp(IRLOp):
    """
    Topological modification.

    Semantics:
    - reads MUST contain exactly one body
    - writes MAY:
        - overwrite the same body (in-place semantic replacement)
        - or create a new derived body
    - topo MUST be provided (explicit target)
    """

    category: IRLOpCategory = field(
        default=IRLOpCategory.FEATURE, init=False
    )


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------

class BooleanKind(str, Enum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


@dataclass(frozen=True)
class BooleanOp(IRLOp):
    """
    Boolean topology operation.

    Semantics:
    - reads MUST contain >= 2 bodies
    - writes MUST be a single resulting body
    """

    category: IRLOpCategory = field(
        default=IRLOpCategory.BOOLEAN, init=False
    )

    kind: BooleanKind = BooleanKind.UNION


# ---------------------------------------------------------------------------
# Export operations (STEP, STL, etc.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportOp(IRLOp):
    """
    Serialization only.

    Semantics:
    - reads MUST contain exactly one body
    - writes MUST be None
    - executor MUST NOT mutate body registry
    """

    category: IRLOpCategory = field(
        default=IRLOpCategory.EXPORT, init=False
    )


# ---------------------------------------------------------------------------
# IRL Model (entire lowered program)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRLModel:
    """
    Fully lowered, executable IRL program.

    Execution order is strictly linear and deterministic.
    No reordering is allowed at executor level.
    """

    ops: List[IRLOp]
