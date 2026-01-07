"""
Kernel grade IRL Executor
=========================

Responsibilities:
- Maintain explicit body registry
- Enforce IRL semantic contract
- Dispatch ops to kernel adapters
- Guarantee determinism

Non-responsibilities:
- No CadQuery workplanes
- No implicit state
- No geometry logic
"""

from __future__ import annotations

from typing import Dict

from lexica.irl.contract import (
    IRLModel,
    IRLOp,
    IRLOpCategory,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    BodyID,
)

# Kernel adapter imports (to be implemented next)
from lexica.cad_engine.adapters.primitive_adapter import execute_primitive
from lexica.cad_engine.adapters.feature_adapter import execute_feature
from lexica.cad_engine.adapters.boolean_adapter import execute_boolean
from lexica.cad_engine.adapters.export_adapter import execute_export
from lexica.cad_engine.adapters.transform_adapter import execute_transform
# from lexica.cad_engine.topology.resolver import resolve_face



class ExecutorError(Exception):
    pass


class BodyRegistry:
    """
    Explicit body registry.

    Maps BodyID -> TopoDS_Shape
    """

    def __init__(self) -> None:
        self._bodies: Dict[BodyID, object] = {}

    def kill(self, body_id: BodyID) -> None:
        if body_id not in self._bodies:
            raise ExecutorError(f"Cannot kill non-existent body '{body_id}'")
        del self._bodies[body_id]

    # ----------------------------
    # Access
    # ----------------------------

    def get(self, body_id: BodyID):
        if body_id not in self._bodies:
            raise ExecutorError(f"Body '{body_id}' does not exist")
        return self._bodies[body_id]

    def set(self, body_id: BodyID, shape) -> None:
        self._bodies[body_id] = shape

    def exists(self, body_id: BodyID) -> bool:
        return body_id in self._bodies

    # ----------------------------
    # Validation helpers
    # ----------------------------

    def require_all(self, body_ids: list[BodyID]) -> None:
        for bid in body_ids:
            if bid not in self._bodies:
                raise ExecutorError(f"Required body '{bid}' not found")

    def forbid_overwrite(self, body_id: BodyID) -> None:
        if body_id in self._bodies:
            raise ExecutorError(
                f"Body '{body_id}' already exists (explicit overwrite required)"
            )


class IRLExecutor:
    """
    Deterministic executor for IRL programs.
    """

    def __init__(self) -> None:
        self.registry = BodyRegistry()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, model: IRLModel) -> None:
        """
        Execute IRL ops in order.

        Execution is strictly linear.
        Reordering is forbidden.
        """
        for op in model.ops:
            self._execute_op(op)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _execute_op(self, op: IRLOp) -> None:
        if op.category == IRLOpCategory.PRIMITIVE:
            self._exec_primitive(op)

        elif op.category == IRLOpCategory.FEATURE:
            self._exec_feature(op)

        elif op.category == IRLOpCategory.BOOLEAN:
            self._exec_boolean(op)
        
        elif op.category == IRLOpCategory.TRANSFORM:
            self._exec_transform(op)

        elif op.category == IRLOpCategory.EXPORT:
            self._exec_export(op)

        else:
            raise ExecutorError(f"Unknown IRL op category: {op.category}")

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    def _exec_primitive(self, op: PrimitiveOp) -> None:
        if op.reads:
            raise ExecutorError(
                f"Primitive op '{op.op_id}' must not read bodies"
            )
        if not op.writes:
            raise ExecutorError(
                f"Primitive op '{op.op_id}' must declare a write body"
            )

        self.registry.forbid_overwrite(op.writes)

        shape = execute_primitive(op)
        self.registry.set(op.writes, shape)

    def _exec_feature(self, op: FeatureOp) -> None:
        if len(op.reads) != 1:
            raise ExecutorError(
                f"Feature op '{op.op_id}' must read exactly one body"
            )
        if not op.writes:
            raise ExecutorError(
                f"Feature op '{op.op_id}' must declare a write body"
            )

        # Topology required for topo-dependent features
        kind = op.params.get("kind")
        if kind in ("fillet", "chamfer") and op.topo is None:
            raise ExecutorError(
                f"Feature op '{op.op_id}' of kind '{kind}' "
                f"requires topology selection"
            )

        self.registry.require_all(op.reads)

        src_id = op.reads[0]
        input_shape = self.registry.get(src_id)
        output_shape = execute_feature(op, input_shape)

        if op.overwrite:
            self.registry.kill(src_id)

        self.registry.set(op.writes, output_shape)

    def _exec_boolean(self, op: BooleanOp) -> None:
        if len(op.reads) < 2:
            raise ExecutorError(
                f"Boolean op '{op.op_id}' must read >= 2 bodies"
            )
        if not op.writes:
            raise ExecutorError(
                f"Boolean op '{op.op_id}' must declare a write body"
            )

        self.registry.require_all(op.reads)

        input_shapes = [self.registry.get(bid) for bid in op.reads]

        try:
            result_shape = execute_boolean(op, input_shapes)
        except Exception as e:
            raise ExecutorError(
                f"Boolean op '{op.op_id}' failed: {e}"
            )

        # Kill all inputs only after success
        for bid in op.reads:
            self.registry.kill(bid)

        self.registry.set(op.writes, result_shape)
    
    def _exec_transform(self, op: TransformOp) -> None:
        if len(op.reads) != 1:
            raise ExecutorError(
                f"Transform op '{op.op_id}' must read exactly one body"
            )
        if not op.writes:
            raise ExecutorError(
                f"Transform op '{op.op_id}' must declare a write body"
            )

        self.registry.require_all(op.reads)

        src_id = op.reads[0]
        input_shape = self.registry.get(src_id)

        # --------------------------
        # Execute (no side effects)
        # --------------------------
        try:
            output_shape = execute_transform(op, input_shape)
        except Exception as e:
            raise ExecutorError(
                f"Transform op '{op.op_id}' failed: {e}"
            )

        # --------------------------
        # Transform creates a new body; source remains LIVE
        # --------------------------
        self.registry.set(op.writes, output_shape)
    
    # def _resolve_pivot(self, shape, pivot):
    #     face = resolve_face(shape, pivot.face)
    #     bb = face.BoundingBox()

    #     if pivot.origin == "center":
    #         origin = bb.center
    #     elif pivot.origin == "min":
    #         origin = bb.min
    #     else:
    #         origin = bb.max

    #     normal = face.normalAt()
    #     return origin, normal

    def _exec_export(self, op: ExportOp) -> None:
        if len(op.reads) != 1:
            raise ExecutorError(
                f"Export op '{op.op_id}' must read exactly one body"
            )
        if op.writes is not None:
            raise ExecutorError(
                f"Export op '{op.op_id}' must not write a body"
            )

        self.registry.require_all(op.reads)

        shape = self.registry.get(op.reads[0])
        execute_export(op, shape)

