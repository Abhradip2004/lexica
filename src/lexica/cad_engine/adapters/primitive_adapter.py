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
    
    if kind == "sphere":
        return _sphere(op)
    
    if kind == "cone":
        return _cone(op)

    if kind == "torus":
        return _torus(op)

    raise PrimitiveAdapterError(f"Unknown primitive kind: {kind}")


def _box(op: PrimitiveOp) -> cq.Workplane:
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

def _cylinder(op: PrimitiveOp) -> cq.Workplane:
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

def _sphere(op: PrimitiveOp) -> cq.Workplane:
    """
    Create a sphere primitive.

    IR v1 contract:
    - r --> radius

    Validation guarantees presence of r.
    """

    try:
        r = op.params["r"]
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing sphere param: {e}. "
            "Required param is 'r'."
        )

    wp = cq.Workplane("XY").sphere(r)
    return wp.val()

    
def _cone(op: PrimitiveOp) -> cq.Workplane:
    """
    Create a cone primitive.

    IR v1 contract:
    - r1 --> base radius (bottom)
    - r2 --> top radius
    - z  --> height along Z axis

    If r2 is 0, it is a true cone.
    Validation should guarantee presence of r1, r2, z.
    """

    try:
        r1 = op.params["r1"]  # base radius
        r2 = op.params["r2"]  # top radius
        z  = op.params["z"]   # height
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing cone param: {e}. "
            "Required params are 'r1', 'r2', and 'z'."
        )

    # CadQuery supports cone(height, radius1, radius2)
    wp = cq.Workplane("XY").cone(z, r1, r2)
    return wp.val()

    
def _torus(op: PrimitiveOp) -> cq.Workplane:
    """
    Create a torus primitive (donut).

    IR v1 contract:
    - R --> major radius (distance from center to tube center)
    - r --> minor radius (tube radius)

    Validation should guarantee presence of R and r.
    """

    try:
        R = op.params["R"]  # major radius
        r = op.params["r"]  # minor radius
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing torus param: {e}. "
            "Required params are 'R' and 'r'."
        )

    # Guaranteed method: revolve a circle around Z axis
    wp = cq.Workplane("XY").center(R, 0).circle(r).revolve(360)
    return wp.val()
