from __future__ import annotations
from typing import List
from lexica.irl.contract import (
    IRModel,
    IRPrimitive,
    IRLOperation,
    IRLProgram,
    IRToIRLCompiler,
)

class DefaultIRToIRLCompiler(IRToIRLCompiler):
    """
    Lexica default lowering pass:
    IR  -> ordered IRL operations
    """

    def __init__(self, backend: str = "cadquery"):
        self.backend = backend

    def compile(self, ir: IRModel) -> IRLProgram:
        operations: List[IRLOperation] = []

        for primitive in ir.primitives:
            operations.extend(self._lower_primitive(primitive))

        return IRLProgram(
            operations=operations,
            backend=self.backend,
        )

#--------------------
# Lowering 
#--------------------

    def _lower_primitive(self, primitive: IRPrimitive) -> List[IRLOperation]:
        ops: List[IRLOperation] = []

        # Create solid
        ops.append(
            IRLOperation(
                op=f"create_{primitive.type}",
                args=primitive.params,
            )
        )

        # Apply transforms in order
        for transform in primitive.transforms:
            ops.append(
                IRLOperation(
                    op=transform["type"],
                    args=transform.get("params", {}),
                )
            )

        return ops
