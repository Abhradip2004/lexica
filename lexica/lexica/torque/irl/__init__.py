"""
IRL (Intermediate Representation – Lowered)

IRL is the executable, kernel-facing representation used by Torque.

Properties:
- Explicit body identities
- Explicit read/write sets
- Deterministic execution order
- No implicit CAD state
- Kernel-grade semantics

IRL is produced by lowering Lexica IR and is validated
before any kernel execution.
"""

from .contract import (
    IRLModel,
    IRLOp,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    TransformOp,
    ExportOp,
    IRLOpCategory,
)

from .validation import validate_irl, IRLValidationError
from .ir_to_irl import lower_ir_to_irl, LoweringError
