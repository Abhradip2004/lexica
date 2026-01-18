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
    Create a cone/frustum using OCC solid primitive.

    Params:
      - r1: base radius
      - r2: top radius
      - z:  height
    """
    try:
        r1 = float(op.params["r1"])
        r2 = float(op.params["r2"])
        z  = float(op.params["z"])
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing cone param: {e}. Required params are 'r1', 'r2', and 'z'."
        )

    if z <= 0:
        raise PrimitiveAdapterError("Cone height z must be > 0")
    if r1 < 0 or r2 < 0:
        raise PrimitiveAdapterError("Cone radii r1/r2 must be >= 0")
    if r1 == 0 and r2 == 0:
        raise PrimitiveAdapterError("Invalid cone: r1 and r2 both 0")

    solid = cq.Solid.makeCone(r1, r2, z)
    return cq.Workplane("XY").newObject([solid]).val()




    
def _torus(op: PrimitiveOp) -> cq.Workplane:
    """
    Create a solid torus using OCC solid primitive.

    Params:
      - R: major radius
      - r: minor radius
    """
    try:
        R = float(op.params["R"])
        r = float(op.params["r"])
    except KeyError as e:
        raise PrimitiveAdapterError(
            f"Missing torus param: {e}. Required params are 'R' and 'r'."
        )

    if R <= 0 or r <= 0:
        raise PrimitiveAdapterError("Torus radii must be > 0")
    if r >= R:
        raise PrimitiveAdapterError("Invalid torus: must satisfy r < R")

    solid = cq.Solid.makeTorus(R, r)
    return cq.Workplane("XY").newObject([solid]).val()


