from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any

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

If an operation cannot be expressed here, it is NOT allowed in Lexica.

Body Lifecycle Semantics (NON-NEGOTIABLE)
=======================================

Lexica enforces explicit, deterministic body lifecycle rules.

Definitions
-----------
- A Body is an immutable topological solid identified by a BodyID.
- Bodies are never mutated in-place.
- Bodies are replaced by new bodies via operations.

Lifecycle States
----------------
- LIVE:
    - Body exists and may be read by operations.
- DEAD:
    - Body has been consumed or replaced.
    - Reading a DEAD body is a hard error.

Operation Semantics
-------------------
PrimitiveOp:
    - Reads: none
    - Writes: exactly one new LIVE body
    - Does not kill any body

FeatureOp:
    - Reads: exactly one LIVE body
    - Writes: exactly one body
    - The input body is considered DEAD after the operation (replacement semantics)

BooleanOp:
    - Reads: >= 2 LIVE bodies
    - Writes: exactly one body
    - ALL input bodies become DEAD after the operation

ExportOp:
    - Reads: exactly one LIVE body
    - Writes: none
    - Does not modify lifecycle state

Illegal States (Hard Errors)
----------------------------
- Reading a DEAD body
- Reusing a consumed boolean operand
- Exporting a DEAD body
- Implicitly mutating bodies
"""





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
    TRANSFORM = "transform"
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
    Declarative topology selection.

    This expresses *intent*, not kernel object identity.
    Resolution must be deterministic and stateless.
    """

    target: TopoTarget

    rule: Literal[
        # --------------------------
        # Face rules (stable)
        # --------------------------
        "normal",        # face normal aligned with axis (+Z, -X, etc.)
        "min",           # extremal face at minimum axis value
        "max",           # extremal face at maximum axis value

        # --------------------------
        # Edge rules (restricted)
        # --------------------------
        "parallel",      # edges parallel to axis
        "length_gt",     # edges with length > value
        "length_lt",     # edges with length < value
    ]

    # Meaning depends on rule:
    # - normal   → "+Z", "-X", etc.
    # - min/max  → "X", "Y", or "Z"
    # - parallel → "X", "Y", or "Z"
    # - length_*→ numeric threshold
    value: Optional[object] = None


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

    category: IRLOpCategory = IRLOpCategory.PRIMITIVE


# ---------------------------------------------------------------------------
# Feature operations (fillet, chamfer, shell, draft, etc.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureOp(IRLOp):
    """
    Topological modification.

    Semantics:
    - reads MUST contain exactly one LIVE body
    - writes MUST produce exactly one body
    - overwrite=True  → input body is replaced (input becomes DEAD)
    - overwrite=False → fork (input stays LIVE, new body created)
    """

    overwrite: bool = True

    category: IRLOpCategory = IRLOpCategory.FEATURE


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

    category: IRLOpCategory = IRLOpCategory.BOOLEAN

    kind: BooleanKind = BooleanKind.UNION

#----------------------------------------------------------------------------
# Transform
#----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformOp(IRLOp):
    category: IRLOpCategory = IRLOpCategory.TRANSFORM


@dataclass(frozen=True)
class FaceSelector:
    """
    Declarative face selector.

    Examples:
    - normal = "+Z"
    - normal = "-Y"
    """
    normal: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    index: Optional[int] = None


@dataclass(frozen=True)
class Pivot:
    face: Union[FaceSelector, EdgeSelector]  # extend
    origin: Literal["center","min","max"] = "center"
    vertex: Optional[VertexSelector] = None  # precise pt
    
@dataclass(frozen=True)
class EdgeSelector:
    """Declarative edge selector."""
    normal: Literal["+X","-X","+Y","-Y","+Z","-Z"]  # edge normal proj
    index: Optional[int] = None
    length_gt: Optional[float] = None  # filter edges > length
    length_lt: Optional[float] = None

@dataclass(frozen=True)
class VertexSelector:
    """Declarative vertex selector."""
    edge: EdgeSelector  # vertex on edge
    extremum: Literal["min","max"] = "min"  # along edge param


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

    category: IRLOpCategory = IRLOpCategory.EXPORT


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
