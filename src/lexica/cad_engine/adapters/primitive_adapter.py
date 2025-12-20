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
    try:
        length = op.params["length"]
        width = op.params["width"]
        height = op.params["height"]
    except KeyError as e:
        raise PrimitiveAdapterError(f"Missing box param: {e}")

    wp = cq.Workplane("XY").box(length, width, height)
    return wp.val()


def _cylinder(op: PrimitiveOp):
    try:
        radius = op.params["radius"]
        height = op.params["height"]
    except KeyError as e:
        raise PrimitiveAdapterError(f"Missing cylinder param: {e}")

    wp = cq.Workplane("XY").circle(radius).extrude(height)
    return wp.val()
