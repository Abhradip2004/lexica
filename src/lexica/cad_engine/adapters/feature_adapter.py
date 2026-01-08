from __future__ import annotations

import math

import cadquery as cq
from cadquery import Solid

from lexica.irl.contract import FeatureOp, TopoPredicate
from lexica.cad_engine.topology.resolver import resolve_face
from lexica.cad_engine.topology.resolver import resolve_edge
from lexica.irl.contract import TopoTarget

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

def _fillet(op: FeatureOp, shape):
    """
    Apply fillet using Topology v2 intent.

    Supported topology:
    - EDGE  → fillet a specific resolved edge
    - FACE  → fillet all edges adjacent to a resolved face
    """

    radius = op.params.get("radius")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise FeatureAdapterError("Fillet radius must be positive")

    if op.topo is None:
        raise FeatureAdapterError("Fillet requires topology selection")

    # --------------------------
    # Edge-based fillet
    # --------------------------
    if op.topo.target == TopoTarget.EDGE:
        edge = resolve_edge(shape, op.topo)

        wp = cq.Workplane(obj=shape).newObject([edge])
        return wp.fillet(radius).val()

    # --------------------------
    # Face-based fillet
    # --------------------------
    elif op.topo.target == TopoTarget.FACE:
        face = resolve_face(shape, op.topo)

        face_edges = list(face.Edges())
        if not face_edges:
            raise FeatureAdapterError("Selected face has no edges to fillet")

        wp = cq.Workplane(obj=shape).newObject(face_edges)
        return wp.fillet(radius).val()

    else:
        raise FeatureAdapterError(
            f"Unsupported topology target '{op.topo.target}' for fillet"
        )



def _chamfer(op: FeatureOp, shape):
    """
    Apply chamfer using Topology v2 intent.

    Supported topology:
    - EDGE  → chamfer a specific resolved edge
    - FACE  → chamfer all edges adjacent to a resolved face
    """

    distance = op.params.get("distance")
    if not isinstance(distance, (int, float)) or distance <= 0:
        raise FeatureAdapterError("Chamfer distance must be positive")

    if op.topo is None:
        raise FeatureAdapterError("Chamfer requires topology selection")

    # --------------------------
    # Edge-based chamfer
    # --------------------------
    if op.topo.target == TopoTarget.EDGE:
        edge = resolve_edge(shape, op.topo)

        wp = cq.Workplane(obj=shape).newObject([edge])
        return wp.chamfer(distance).val()

    # --------------------------
    # Face-based chamfer
    # --------------------------
    elif op.topo.target == TopoTarget.FACE:
        face = resolve_face(shape, op.topo)

        # Extract all edges of this face
        face_edges = list(face.Edges())
        if not face_edges:
            raise FeatureAdapterError("Selected face has no edges to chamfer")

        wp = cq.Workplane(obj=shape).newObject(face_edges)
        return wp.chamfer(distance).val()

    else:
        raise FeatureAdapterError(
            f"Unsupported topology target '{op.topo.target}' for chamfer"
        )




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

    face_sel = FaceSelector(**params["face"])
    cx, cy = params.get("center", (0, 0))

    dir_map = {
        "+X": ">X", "-X": "<X",
        "+Y": ">Y", "-Y": "<Y",
        "+Z": ">Z", "-Z": "<Z",
    }
    dir_str = dir_map[face_sel.normal]

    # -------------------------------------------------
    # SINGLE workplane 
    # -------------------------------------------------
    wp = (
        cq.Workplane(obj=shape)
        .faces(dir_str)
        .workplane()
        .center(cx, cy)
    )

    # -------------------------------------------------
    # Base hole
    # -------------------------------------------------
    if through_all:
        wp = wp.hole(diameter)
    else:
        wp = wp.hole(diameter, depth)

    # -------------------------------------------------
    # Counterbore (same workplane!)
    # -------------------------------------------------
    if counterbore:
        wp = wp.hole(
            counterbore["diameter"],
            counterbore["depth"],
        )

    # -------------------------------------------------
    # Countersink (same workplane!)
    # -------------------------------------------------
    if countersink:
        cs_d = countersink["diameter"]
        cs_angle = countersink["angle_deg"]

        radius_diff = (cs_d - diameter) / 2.0
        cone_height = radius_diff / math.tan(math.radians(cs_angle / 2.0))

        wp = wp.cone(
            height=cone_height,
            r1=cs_d / 2.0,
            r2=diameter / 2.0,
        ).cut(wp)

    return wp.val()

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
