from __future__ import annotations

import math

import cadquery as cq
from cadquery import Solid

from lexica.irl.contract import FeatureOp, TopoPredicate


class FeatureAdapterError(Exception):
    pass


def execute_feature(op: FeatureOp, input_shape):
    """
    Execute a feature op on a single body.
    """

    kind = op.params.get("kind")

    if kind == "fillet":
        return _fillet(op, input_shape)

    if kind == "chamfer":
        return _chamfer(op, input_shape)

    if kind == "translate":
        return _translate(op, input_shape)

    if kind == "rotate":
        return _rotate(op, input_shape)
    
    if kind == "shell":
        return _shell(op, input_shape)
    
    elif kind == "hole":
        return _hole(op, input_shape)

    raise FeatureAdapterError(f"Unknown feature kind: {kind}")


# ---------------------------
# Feature implementations
# ---------------------------

def _fillet(op: FeatureOp, shape) -> cq.Workplane:
    radius = op.params.get("radius")
    if radius is None:
        raise FeatureAdapterError("Fillet requires radius")

    wp = cq.Workplane(obj=shape)

    edges = _select_edges(wp, op.topo)
    result = edges.fillet(radius)

    return result.val()


def _chamfer(op: FeatureOp, shape) -> cq.Workplane:
    distance = op.params.get("distance")
    if distance is None:
        raise FeatureAdapterError("Chamfer requires distance")

    wp = cq.Workplane(obj=shape)

    edges = _select_edges(wp, op.topo)
    result = edges.chamfer(distance)

    return result.val()

def _translate(op: FeatureOp, shape) -> cq.Workplane:
    dx = op.params.get("dx", 0.0)
    dy = op.params.get("dy", 0.0)
    dz = op.params.get("dz", 0.0)

    wp = cq.Workplane(obj=shape)
    result = wp.translate((dx, dy, dz))

    return result.val()

def _rotate(op: FeatureOp, shape) -> cq.Workplane:
    axis = op.params.get("axis")
    angle = op.params.get("angle_deg")

    if axis not in ("x", "y", "z"):
        raise FeatureAdapterError("Rotate axis must be x, y, or z")

    if angle is None:
        raise FeatureAdapterError("Rotate requires angle_deg")

    axis_map = {
        "x": ((0, 0, 0), (1, 0, 0)),
        "y": ((0, 0, 0), (0, 1, 0)),
        "z": ((0, 0, 0), (0, 0, 1)),
    }

    p1, p2 = axis_map[axis]

    wp = cq.Workplane(obj=shape)
    result = wp.rotate(p1, p2, angle)

    return result.val()

def _shell(op: FeatureOp, shape) -> cq.Workplane:
    thickness = op.params.get("thickness")

    if thickness is None:
        raise FeatureAdapterError("Shell requires 'thickness' parameter")

    if not isinstance(thickness, (int, float)):
        raise FeatureAdapterError("Shell thickness must be a number")

    if thickness <= 0:
        raise FeatureAdapterError("Shell thickness must be positive")

    try:
        wp = cq.Workplane(obj=shape)

        # Inward shell (engineering default)
        # Negative thickness preserves external dimensions
        result = wp.faces().shell(-float(thickness))

    except Exception as e:
        # OCC / CadQuery throws for impossible shells
        raise FeatureAdapterError(
            f"Shell operation failed (thickness={thickness}): {e}"
        )

    # CadQuery returns a Workplane; extract the shape
    return result.val()

def _hole(op: FeatureOp, shape):
    params = op.params

    diameter = params.get("diameter")
    depth = params.get("depth")
    through_all = params.get("through_all", False)
    counterbore = params.get("counterbore")
    countersink = params.get("countersink")

    # --------------------------
    # Basic validation
    # --------------------------
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise FeatureAdapterError("Hole diameter must be positive number")

    if depth is not None:
        if not isinstance(depth, (int, float)) or depth <= 0:
            raise FeatureAdapterError("Hole depth must be positive number")

    # --------------------------
    # Determine hole depth
    # --------------------------
    try:
        bb = shape.BoundingBox()
    except Exception:
        raise FeatureAdapterError("Failed to compute shape bounding box")

    if through_all:
        hole_height = bb.zlen + 2.0
    else:
        hole_height = float(depth)

    try:
        wp = cq.Workplane(obj=shape)

        # --------------------------
        # Base hole
        # --------------------------
        cutter = (
            cq.Workplane("XY")
            .circle(diameter / 2.0)
            .extrude(hole_height)
        )

        result = wp.cut(cutter)

        # --------------------------
        # Counterbore
        # --------------------------
        if counterbore:
            cb_diam = counterbore.get("diameter")
            cb_depth = counterbore.get("depth")

            if cb_diam <= diameter:
                raise FeatureAdapterError(
                    "Counterbore diameter must be larger than hole diameter"
                )

            if cb_depth <= 0:
                raise FeatureAdapterError(
                    "Counterbore depth must be positive"
                )

            cb_cutter = (
                cq.Workplane("XY")
                .circle(cb_diam / 2.0)
                .extrude(cb_depth)
            )

            result = result.cut(cb_cutter)

        # --------------------------
        # Countersink
        # --------------------------
        if countersink:
            cs_diam = countersink.get("diameter")
            cs_angle = countersink.get("angle_deg")

            if cs_diam <= diameter:
                raise FeatureAdapterError(
                    "Countersink diameter must be larger than hole diameter"
                )

            if cs_angle <= 0 or cs_angle >= 180:
                raise FeatureAdapterError(
                    "Countersink angle must be between 0 and 180 degrees"
                )

            # Compute cone height from angle
            radius_diff = (cs_diam - diameter) / 2.0
            cone_height = radius_diff / math.tan(
                math.radians(cs_angle / 2.0)
            )

            cs_cone = Solid.makeCone(
                cs_diam / 2.0,      # top radius
                diameter / 2.0,     # bottom radius
                cone_height,
            )

            # Position cone at top face (Z=0 assumption for v1)
            cs_wp = cq.Workplane(obj=cs_cone)

            result = result.cut(cs_wp)

    except FeatureAdapterError:
        raise
    except Exception as e:
        raise FeatureAdapterError(
            f"Hole operation failed: {e}"
        )

    return result.val()


# ---------------------------
# Topology selection (Level 0)
# ---------------------------

def _select_edges(wp: cq.Workplane, topo: TopoPredicate) -> cq.Workplane:
    if topo.target.value != "edge":
        raise FeatureAdapterError(
            f"Unsupported topo target: {topo.target}"
        )

    rule = topo.rule

    if rule == "all":
        sel = wp.edges()

    elif rule == "convex":
        sel = wp.edges("|Z")  # placeholder, deterministic

    elif rule == "by_length_gt":
        sel = wp.edges().filter(lambda e: e.Length() > topo.value)

    elif rule == "by_length_lt":
        sel = wp.edges().filter(lambda e: e.Length() < topo.value)

    else:
        raise FeatureAdapterError(f"Unknown topo rule: {rule}")

    # --------------------------
    # HARD GUARANTEE: non-empty
    # --------------------------
    objs = sel.objects
    if not objs:
        raise FeatureAdapterError(
            f"Topology selection returned 0 edges "
            f"(rule='{rule}', value={topo.value})"
        )

    return sel
