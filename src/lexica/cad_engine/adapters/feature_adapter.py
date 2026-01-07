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

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    diameter = params.get("diameter")
    depth = params.get("depth")
    through_all = params.get("through_all", False)
    counterbore = params.get("counterbore")
    countersink = params.get("countersink")

    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise FeatureAdapterError("Hole diameter must be a positive number")

    if not through_all:
        if depth is None or not isinstance(depth, (int, float)) or depth <= 0:
            raise FeatureAdapterError("Blind hole requires positive depth")

    if "face" not in params:
        raise FeatureAdapterError("Hole requires face selector")

    # -------------------------------------------------
    # Topology v1 — FACE SELECTOR (not face object)
    # -------------------------------------------------
    face_sel = FaceSelector(**params["face"])
    resolved = resolve_face(shape, face_sel)

    cx, cy = params.get("center", (0, 0))

    # -------------------------------------------------
    # Base hole
    # -------------------------------------------------
    wp = cq.Workplane(obj=shape)

    if isinstance(face_sel, FaceSelector):
        dir_str = {
            "+X": ">X",
            "-X": "<X",
            "+Y": ">Y",
            "-Y": "<Y",
            "+Z": ">Z",
            "-Z": "<Z",
        }[face_sel.normal]

        wp = wp.faces(dir_str).workplane()
    else:
        raise FeatureAdapterError("Hole requires a FaceSelector")

    if through_all:
        result = wp.hole(diameter)
    else:
        result = wp.hole(diameter, depth)

    # -------------------------------------------------
    # Counterbore
    # -------------------------------------------------
    if counterbore:
        cb_diam = counterbore.get("diameter")
        cb_depth = counterbore.get("depth")

        if not isinstance(cb_diam, (int, float)) or cb_diam <= diameter:
            raise FeatureAdapterError(
                "Counterbore diameter must be larger than hole diameter"
            )

        if not isinstance(cb_depth, (int, float)) or cb_depth <= 0:
            raise FeatureAdapterError("Counterbore depth must be positive")

        result = (
            cq.Workplane(obj=result)
            .faces(face_selector)
            .workplane()
            .center(cx, cy)
            .hole(cb_diam, cb_depth)
        )

    # -------------------------------------------------
    # Countersink
    # -------------------------------------------------
    if countersink:
        cs_diam = countersink.get("diameter")
        cs_angle = countersink.get("angle_deg")

        if not isinstance(cs_diam, (int, float)) or cs_diam <= diameter:
            raise FeatureAdapterError(
                "Countersink diameter must be larger than hole diameter"
            )

        if not isinstance(cs_angle, (int, float)) or not (0 < cs_angle < 180):
            raise FeatureAdapterError(
                "Countersink angle must be between 0 and 180 degrees"
            )

        radius_diff = (cs_diam - diameter) / 2.0
        cone_height = radius_diff / math.tan(math.radians(cs_angle / 2.0))

        result = (
            cq.Workplane(obj=result)
            .faces(face_selector)
            .workplane()
            .center(cx, cy)
            .cone(
                height=cone_height,
                r1=cs_diam / 2.0,
                r2=diameter / 2.0,
            )
            .cut(result)
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
