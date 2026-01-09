from __future__ import annotations

import cadquery as cq

from lexica.irl.contract import PrimitiveOp


class PrimitiveAdapterError(Exception):
    pass


def execute_primitive(op: PrimitiveOp):
    """
    Execute a primitive IRL op.

    Returns:
        TopoDS_Shape
    """

    kind = op.params.get("kind")

    if kind == "box":
        return _box(op)

    if kind == "cylinder":
        return _cylinder(op)

    raise PrimitiveAdapterError(f"Unknown primitive kind: {kind}")


def _box(op: PrimitiveOp):
    """
    Create a box primitive.

    IR v1 contract:
    - x --> size along X axis
    - y --> size along Y axis
    - z --> size along Z axis

    Validation guarantees presence of x, y, z.
    """

    try:
        x = op.params["x"]  # X axis
        y = op.params["y"]  # Y axis
        z = op.params["z"]  # Z axis
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing box param: {e}. "
            "Required params are 'x', 'y', 'z'."
        )

    wp = cq.Workplane("XY").box(x, y, z)
    return wp.val()

def _cylinder(op: PrimitiveOp):
    """
    Create a cylinder primitive.

    IR v1 contract:
    - r --> radius in XY plane
    - z --> height along Z axis

    Validation guarantees presence of r and z.
    """

    try:
        r = op.params["r"]   # radius
        z = op.params["z"]   # height along Z
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing cylinder param: {e}. "
            "Required params are 'r' and 'z'."
        )

    wp = cq.Workplane("XY").circle(r).extrude(z)
    return wp.val()

