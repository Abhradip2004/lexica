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

def resolve_face(shape: cq.Solid, selector: FaceSelector) -> cq.Face:
    """
    Resolve FaceSelector → single TopoDS_Face.

    Args:
        shape: Input solid body
        selector: Declarative selector (normal + optional index)

    Raises:
        ValueError: Invalid normal / no matches / ambiguous / bad index

    Usage: FeatureOp.hole on "+Z[1]" (2nd top face)
    """
    # WRAP SHAPE FOR SELECTION
    wp = cq.Workplane(obj=shape)
    
    # LOOKUP DIRECTION SELECTOR
    dir_str = NORMAL_MAP.get(selector.normal)
    if dir_str is None:
        valid = "', '".join(NORMAL_MAP)
        raise ValueError(f"Invalid normal '{selector.normal}'; use [{valid}]")
    
    # FILTER FACES BY NORMAL DIRECTION
    # .vals() → List[TopoDS_Face]; empty if no match
    faces = wp.faces(dir_str).vals()
    if not faces:
        raise ValueError(f"No faces match normal '{selector.normal}' on shape")
    
    # DISAMBIGUATE VIA INDEX
    if selector.index is None:
        if len(faces) > 1:
            raise ValueError(
                f"Ambiguous {len(faces)} faces for '{selector.normal}'; "
                f"specify index (0-{len(faces)-1)"
            )
        return faces[0]  # Unique match
    
    # INDEXED ACCESS (0-based)
    if selector.index < 0 or selector.index >= len(faces):
        raise ValueError(
            f"Index {selector.index} out-of-bounds for '{selector.normal}' "
            f"(valid: 0-{len(faces)-1}, found {len(faces)} faces)"
        )
    
    return faces[selector.index]  # Guaranteed single Face

def resolve_edge(shape: cq.Solid, selector: EdgeSelector) -> cq.Edge:
    """
    EdgeSelector → single Edge (proj normal + length filter).
    
    e.g., longest vertical edges: normal="+Z", length_gt=10
    """
    wp = cq.Workplane(obj=shape)
    
    # Edges perpendicular to normal (common CQ: "|Z" vertical edges)
    dir_str = NORMAL_MAP.get(selector.normal.replace("+", "|").replace("-", "|"))  # +Z/-Z → |Z
    if dir_str is None:
        raise ValueError(f"Invalid edge normal '{selector.normal}'")
    
    sel = wp.edges(dir_str)
    
    # LENGTH FILTERS (deterministic)
    if selector.length_gt is not None:
        sel = sel.filter(lambda e: e.Length() > selector.length_gt)
    if selector.length_lt is not None:
        sel = sel.filter(lambda e: e.Length() < selector.length_lt)
    
    edges = sel.vals()
    if not edges:
        raise ValueError(f"No edges match {selector}")
    
    idx = selector.index or 0
    if idx < 0 or idx >= len(edges):
        raise ValueError(f"Edge idx {idx} out-of-bounds (0-{len(edges)-1})")
    
    return edges[idx]

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
