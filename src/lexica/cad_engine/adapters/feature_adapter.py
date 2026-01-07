from __future__ import annotations

import math

import cadquery as cq
from cadquery import Solid

from lexica.irl.contract import FeatureOp, TopoPredicate
from lexica.cad_engine.topology.resolver import resolve_face
from lexica.irl.contract import FaceSelector


# face_sel = FaceSelector(**params["face"])
# face = resolve_face(shape, op.face)
# wp = cq.Workplane(obj=shape).workplane(obj=face)

# wp = wp.center(op.center[0], op.center[1])
# wp = wp.hole(op.diameter)


class FeatureAdapterError(Exception):
    pass


def execute_feature(op: FeatureOp, input_shape):
    """
    Execute a topology changing feature op on a single body.
    """

    kind = op.params.get("kind")

    if kind == "fillet":
        return _fillet(op, input_shape)

    if kind == "chamfer":
        return _chamfer(op, input_shape)

    if kind == "shell":
        return _shell(op, input_shape)

    if kind == "hole":
        return _hole(op, input_shape)

    raise FeatureAdapterError(f"Unknown or invalid feature kind: {kind}")


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

    # --------------------------
    # Validation
    # --------------------------
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise FeatureAdapterError("Hole diameter must be positive")

    if not through_all:
        if not isinstance(depth, (int, float)) or depth <= 0:
            raise FeatureAdapterError("Blind hole requires positive depth")

    if "face" not in params:
        raise FeatureAdapterError("Hole requires face selector")

    if params.get("counterbore") or params.get("countersink"):
        raise FeatureAdapterError(
            "Counterbore / countersink not supported yet"
        )

    # --------------------------
    # Face resolution
    # --------------------------
    face_sel = FaceSelector(**params["face"])
    face = resolve_face(shape, face_sel)

    cx, cy = params.get("center", (0, 0))

    # --------------------------
    # Hole execution
    # --------------------------
    wp = (
        cq.Workplane(obj=shape)
        .faces(face)
        .workplane()
        .center(cx, cy)
    )

    if through_all:
        result = wp.hole(diameter)
    else:
        result = wp.hole(diameter, depth)

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
    
    elif rule == "concave":
        raise FeatureAdapterError(
            "Concave edge selection is not supported yet"
        )

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
