from __future__ import annotations

import cadquery as cq

from lexica.irl.contract import IRLProgram, IRLOperation


class CadQueryBackend:
    """
    Executes IRL operations using CadQuery.
    """

    def __init__(self):
        self.wp = cq.Workplane("XY")

    # -------------------------
    # Entry point
    # -------------------------
    
    def execute(self, program: IRLProgram) -> cq.Workplane:
        for operation in program.operations:
            self._dispatch(operation)

        return self.wp

    # -------------------------
    # Dispatch
    # -------------------------
    
    def _dispatch(self, operation: IRLOperation):
        handler_name = f"_op_{operation.op}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            raise NotImplementedError(
                f"CadQuery backend does not support IRL op '{operation.op}'"
            )

        handler(**operation.args)

    # -------------------------
    # IRL Ops
    # -------------------------

    # ---- creation ----
    def _op_create_box(self, x: float, y: float, z: float):
        self.wp = self.wp.box(x, y, z)

    def _op_create_cylinder(self, radius: float, height: float):
        self.wp = self.wp.cylinder(height, radius)

    def _op_create_sphere(self, radius: float):
        self.wp = self.wp.sphere(radius)

    # ---- transforms ----
    def _op_translate(self, x: float = 0, y: float = 0, z: float = 0):
        self.wp = self.wp.translate((x, y, z))

    def _op_rotate(self, axis: str, angle: float):
        axes = {
            "X": (1, 0, 0),
            "Y": (0, 1, 0),
            "Z": (0, 0, 1),
        }
        
        if axis not in axes:
            raise ValueError(f"Invalid rotation axis: {axis}")

        self.wp = self.wp.rotate((0, 0, 0), axes[axis], angle)

    # ---- edge ops ----
    def _op_fillet(self, radius: float):
        self.wp = self.wp.edges("|Z").fillet(radius)

    def _op_chamfer(self, distance: float):
        self.wp = self.wp.edges("|Z").chamfer(distance)
