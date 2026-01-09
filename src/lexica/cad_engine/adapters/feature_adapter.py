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
    Apply a fillet feature using Topology v2 intent.

    Semantics (IR v1 / Phase 1):
    --------------------------------
    - Fillet ALWAYS requires topology selection
    - EDGE + rule="all"  --> fillet ALL edges of the solid
    - EDGE (other rules) --> fillet resolved edge
    - FACE               --> fillet all edges adjacent to resolved face
    - 'radius' must be a positive scalar
    """

    # -------------------------------------------------
    # Validate parameters
    # -------------------------------------------------
    radius = op.params.get("radius")
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise FeatureAdapterError(
            "Fillet requires positive 'radius'"
        )

    if op.topo is None:
        raise FeatureAdapterError(
            "Fillet requires topology selection"
        )

    # -------------------------------------------------
    # Edge-based fillet
    # -------------------------------------------------
    if op.topo.target == TopoTarget.EDGE:

        # Special case: fillet ALL edges of the solid
        if op.topo.rule == "all":
            all_edges = list(shape.Edges())
            if not all_edges:
                raise FeatureAdapterError(
                    "Shape has no edges to fillet"
                )

            wp = cq.Workplane(obj=shape).newObject(all_edges)
            return wp.fillet(radius).val()

        # Otherwise: resolve a specific edge
        edge = resolve_edge(shape, op.topo)
        if edge is None:
            raise FeatureAdapterError(
                "Topology did not resolve to a valid edge"
            )

        wp = cq.Workplane(obj=shape).newObject([edge])
        return wp.fillet(radius).val()

    # -------------------------------------------------
    # Face-based fillet
    # -------------------------------------------------
    if op.topo.target == TopoTarget.FACE:
        face = resolve_face(shape, op.topo)
        if face is None:
            raise FeatureAdapterError(
                "Topology did not resolve to a valid face"
            )

        face_edges = list(face.Edges())
        if not face_edges:
            raise FeatureAdapterError(
                "Selected face has no edges to fillet"
            )

        wp = cq.Workplane(obj=shape).newObject(face_edges)
        return wp.fillet(radius).val()

    # -------------------------------------------------
    # Unsupported topology target
    # -------------------------------------------------
    raise FeatureAdapterError(
        f"Unsupported topology target '{op.topo.target}' for fillet"
    )



def _chamfer(op: FeatureOp, shape):
    """
    Apply a chamfer feature using Topology v2 intent.

    Semantics (IR v1 / Phase 1):
    --------------------------------
    - Chamfer ALWAYS requires topology selection
    - EDGE target --> chamfer a specific resolved edge
    - FACE target --> chamfer all edges adjacent to a resolved face
    - 'distance' is a positive scalar value

    No default behavior is inferred.
    """

    # -------------------------------------------------
    # Validate parameters
    # -------------------------------------------------
    distance = op.params.get("distance")
    if not isinstance(distance, (int, float)) or distance <= 0:
        raise FeatureAdapterError(
            "Chamfer requires positive 'distance'"
        )

    if op.topo is None:
        raise FeatureAdapterError(
            "Chamfer requires topology selection"
        )

    # -------------------------------------------------
    # Edge-based chamfer
    # -------------------------------------------------
    if op.topo.target == TopoTarget.EDGE:

        # Special case: chamfer ALL edges of the solid
        if op.topo.rule == "all":
            all_edges = list(shape.Edges())
            if not all_edges:
                raise FeatureAdapterError(
                    "Shape has no edges to chamfer"
                )

            wp = cq.Workplane(obj=shape).newObject(all_edges)
            return wp.chamfer(distance).val()

        # Otherwise: resolve a specific edge
        edge = resolve_edge(shape, op.topo)
        if edge is None:
            raise FeatureAdapterError(
                "Topology did not resolve to a valid edge"
            )

        wp = cq.Workplane(obj=shape).newObject([edge])
        return wp.chamfer(distance).val()


    # -------------------------------------------------
    # Face-based chamfer
    # -------------------------------------------------
    if op.topo.target == TopoTarget.FACE:
        face = resolve_face(shape, op.topo)
        if face is None:
            raise FeatureAdapterError(
                "Topology did not resolve to a valid face"
            )

        face_edges = list(face.Edges())
        if not face_edges:
            raise FeatureAdapterError(
                "Selected face has no edges to chamfer"
            )

        wp = cq.Workplane(obj=shape).newObject(face_edges)
        return wp.chamfer(distance).val()

    # -------------------------------------------------
    # Unsupported topology target
    # -------------------------------------------------
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
    """
    Execute a hole feature (MVP / IR v1).

    Semantics (LOCKED for Phase 1):
    --------------------------------
    - Hole is ALWAYS face-based
    - Face selection comes ONLY from IR topology:
        - op.topo --> explicit face selection
        - None default top face (+Z)
    - 'center' means 2D offset on the selected face (cx, cy)
    - 'through_all = True' --> through hole
    - Otherwise --> depth-based blind hole

    IMPORTANT:
    - No body-centered holes in Phase 1
    - No inference from 'center'
    - No mixing of CadQuery APIs
    """

    params = op.params

    # -------------------------------------------------
    # Validate required parameters
    # -------------------------------------------------
    diameter = params.get("diameter")
    if not isinstance(diameter, (int, float)) or diameter <= 0:
        raise FeatureAdapterError("Hole requires positive 'diameter'")

    depth = params.get("depth")
    through_all = bool(params.get("through_all", False))

    if not through_all and depth is None:
        raise FeatureAdapterError(
            "Hole requires 'depth' unless 'through_all' is True"
        )

    # -------------------------------------------------
    # Resolve target face
    # -------------------------------------------------
    if op.topo is not None:
        # Topology-driven face selection (preferred)
        face = resolve_face(shape, op.topo)
        wp = cq.Workplane(obj=shape).newObject([face]).workplane()
    else:
        # Default: top face (+Z)
        wp = (
            cq.Workplane(obj=shape)
            .faces(">Z")
            .workplane()
        )

    # -------------------------------------------------
    # Apply 2D center offset (on face)
    # -------------------------------------------------
    cx, cy = params.get("center", (0.0, 0.0))
    wp = wp.center(cx, cy)

    # -------------------------------------------------
    # Create hole (single, consistent API)
    # -------------------------------------------------
    if through_all:
        wp = wp.hole(diameter)
    else:
        wp = wp.hole(diameter, depth)

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
