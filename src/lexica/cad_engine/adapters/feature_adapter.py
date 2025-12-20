from __future__ import annotations

import cadquery as cq

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

    raise FeatureAdapterError(f"Unknown feature kind: {kind}")


# ---------------------------
# Feature implementations
# ---------------------------

def _fillet(op: FeatureOp, shape):
    radius = op.params.get("radius")
    if radius is None:
        raise FeatureAdapterError("Fillet requires radius")

    wp = cq.Workplane(obj=shape)

    edges = _select_edges(wp, op.topo)
    result = edges.fillet(radius)

    return result.val()


def _chamfer(op: FeatureOp, shape):
    distance = op.params.get("distance")
    if distance is None:
        raise FeatureAdapterError("Chamfer requires distance")

    wp = cq.Workplane(obj=shape)

    edges = _select_edges(wp, op.topo)
    result = edges.chamfer(distance)

    return result.val()

def _translate(op: FeatureOp, shape):
    dx = op.params.get("dx", 0.0)
    dy = op.params.get("dy", 0.0)
    dz = op.params.get("dz", 0.0)

    wp = cq.Workplane(obj=shape)
    result = wp.translate((dx, dy, dz))

    return result.val()

def _rotate(op: FeatureOp, shape):
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


# ---------------------------
# Topology selection (Level 0)
# ---------------------------

def _select_edges(wp: cq.Workplane, topo: TopoPredicate):
    if topo.target.value != "edge":
        raise FeatureAdapterError(
            f"Unsupported topo target: {topo.target}"
        )

    rule = topo.rule

    if rule == "all":
        return wp.edges()

    if rule == "convex":
        return wp.edges("|Z")  # placeholder predicate

    if rule == "by_length_gt":
        return wp.edges().filter(lambda e: e.Length() > topo.value)

    if rule == "by_length_lt":
        return wp.edges().filter(lambda e: e.Length() < topo.value)

    raise FeatureAdapterError(f"Unknown topo rule: {rule}")
