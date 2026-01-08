"""
Topology Resolution Module

Responsibilities:
- Map declarative selectors (FaceSelector etc.) → CadQuery topo objects
- Deterministic, exhaustive error handling
- No business logic, no execution

Semantics:
- Single object return (index 0 if unspecified)
- Raises on empty/ambiguous/out-of-bounds
- NORMAL_MAP: CadQuery direction strings (">X" = faces normal +X)
"""

import cadquery as cq
from typing import Union, List

from lexica.irl.contract import (
    FaceSelector, 
    EdgeSelector, 
    VertexSelector,
)
from lexica.irl.contract import TopoPredicate, TopoTarget


# CADQUERY DIRECTION MAPPING
# Key: IRL normal literals ("+X" etc.)
# Value: CQ selector (">X" = faces with normal > +X axis)
NORMAL_MAP: dict[str, str] = {
    "+X": ">X",   # Max X-normal faces
    "-X": "<X",   # Min X-normal faces
    "+Y": ">Y",
    "-Y": "<Y",
    "+Z": ">Z",   # Top faces (engineering default)
    "-Z": "<Z",   # Bottom faces
}

def resolve_face(shape: cq.Solid, selector: FaceSelector | TopoPredicate) -> cq.Face:
    """
    Resolve face topology intent into a single Face.

    Supports:
    - normal-based selection
    - extremal (min / max) selection

    Raises on:
    - no matches
    - ambiguous matches
    """

    wp = cq.Workplane(obj=shape)

    # -------------------------------------------------
    # FaceSelector (legacy, still supported)
    # -------------------------------------------------
    if isinstance(selector, FaceSelector):
        NORMAL_MAP = {
            "+X": ">X", "-X": "<X",
            "+Y": ">Y", "-Y": "<Y",
            "+Z": ">Z", "-Z": "<Z",
        }

        dir_str = NORMAL_MAP.get(selector.normal)
        if dir_str is None:
            raise ValueError(f"Invalid face normal '{selector.normal}'")

        faces = wp.faces(dir_str).vals()

        if not faces:
            raise ValueError(f"No faces match normal '{selector.normal}'")

        if selector.index is None:
            if len(faces) > 1:
                raise ValueError(
                    f"Ambiguous {len(faces)} faces for normal '{selector.normal}'"
                )
            return faces[0]

        if selector.index < 0 or selector.index >= len(faces):
            raise ValueError(
                f"Face index {selector.index} out of range (0-{len(faces)-1})"
            )

        return faces[selector.index]

    # -------------------------------------------------
    # TopoPredicate (Topology v2)
    # -------------------------------------------------
    if not isinstance(selector, TopoPredicate):
        raise ValueError(f"Unsupported face selector type: {type(selector)}")

    rule = selector.rule
    value = selector.value

    faces = wp.faces().vals()
    if not faces:
        raise ValueError("Shape contains no faces")

    # --------------------------
    # Normal-aligned faces
    # --------------------------
    if rule == "normal":
        if value not in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            raise ValueError(f"Invalid normal value '{value}'")

        NORMAL_MAP = {
            "+X": ">X", "-X": "<X",
            "+Y": ">Y", "-Y": "<Y",
            "+Z": ">Z", "-Z": "<Z",
        }

        matches = wp.faces(NORMAL_MAP[value]).vals()

    # --------------------------
    # Extremal faces
    # --------------------------
    elif rule in ("min", "max"):
        if value not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid axis '{value}' for extremal face")

        key = {
            "X": lambda f: f.Center().x,
            "Y": lambda f: f.Center().y,
            "Z": lambda f: f.Center().z,
        }[value]

        faces_sorted = sorted(faces, key=key)
        extreme_val = key(faces_sorted[0]) if rule == "min" else key(faces_sorted[-1])

        matches = [
            f for f in faces
            if abs(key(f) - extreme_val) < 1e-6
        ]

    else:
        raise ValueError(f"Unsupported face rule '{rule}'")

    if not matches:
        raise ValueError(f"No faces match rule '{rule}' with value '{value}'")

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous face selection: {len(matches)} matches for rule '{rule}'"
        )

    return matches[0]

def resolve_edge(shape: cq.Solid, selector: TopoPredicate) -> cq.Edge:
    """
    Resolve edge topology intent into a single Edge.

    Supports:
    - parallel to axis
    - length filters
    """

    if selector.target != TopoTarget.EDGE:
        raise ValueError("resolve_edge requires EDGE target")

    wp = cq.Workplane(obj=shape)
    edges = wp.edges().vals()

    if not edges:
        raise ValueError("Shape contains no edges")

    rule = selector.rule
    value = selector.value

    # --------------------------
    # Axis-parallel edges
    # --------------------------
    if rule == "parallel":
        if value not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid axis '{value}' for parallel edge")

        axis_vec = {
            "X": cq.Vector(1, 0, 0),
            "Y": cq.Vector(0, 1, 0),
            "Z": cq.Vector(0, 0, 1),
        }[value]

        def is_parallel(edge):
            d = edge.tangentAt(0.5)
            return abs(d.dot(axis_vec)) > 0.99

        matches = [e for e in edges if is_parallel(e)]

        if not matches:
            raise ValueError("No parallel edges found")

        # Deterministic ordering (Topology v2 rule)
        matches = sorted(
            matches,
            key=lambda e: (
                round(e.Center().z, 6),
                round(e.Length(), 6),
            ),
            reverse=True,
        )

        idx = selector.index or 0
        if idx >= len(matches):
            raise ValueError(
                f"Edge index {idx} out of range (0-{len(matches)-1})"
            )

        return matches[idx]

    # --------------------------
    # Length filters
    # --------------------------
    elif rule == "length_gt":
        matches = [e for e in edges if e.Length() > float(value)]

    elif rule == "length_lt":
        matches = [e for e in edges if e.Length() < float(value)]
    
    elif rule == "all":
        # Legacy Topology v1 compatibility:
        # deterministically select the longest edge
        matches = sorted(edges, key=lambda e: e.Length(), reverse=True)

    else:
        raise ValueError(f"Unsupported edge rule '{rule}'")

    if not matches:
        raise ValueError(f"No edges match rule '{rule}'")

    # Legacy Topology v1 compatibility:
    # "all" is allowed to be ambiguous but must be deterministic
    if rule != "all" and len(matches) > 1:
        raise ValueError(
            f"Ambiguous edge selection: {len(matches)} matches for rule '{rule}'"
        )

    return matches[0]


def resolve_vertex(shape: cq.Solid, selector: VertexSelector) -> cq.Vector:
    """VertexSelector → precise Vector (edge endpoint/center)."""
    edge = resolve_edge(shape, selector.edge)
    bb = edge.BoundingBox()
    
    if selector.extremum == "min":
        return cq.Vector(bb.xmin, bb.ymin, bb.zmin)
    elif selector.extremum == "max":
        return cq.Vector(bb.xmax, bb.ymax, bb.zmax)
    elif selector.extremum == "center":
        return cq.Vector(bb.xmid, bb.ymid, bb.zmid)
    else:
        raise ValueError(f"Invalid extremum '{selector.extremum}'")
