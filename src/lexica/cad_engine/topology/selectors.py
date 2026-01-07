import cadquery as cq
from lexica.irl.contract import (
    FaceSelector, EdgeSelector, VertexSelector,
    TopoPredicate,
)

# Shared normal -> CQ selector map 
NORMAL_MAP = {
    "+X": "|X",
    "-X": "|X",  # edges perp to X? Adjust proj logic
    "+Y": "|Y",
    "-Y": "|Y",
    "+Z": "|Z",
    "-Z": "|Z",
}

def resolve_face(shape, selector: FaceSelector) -> cq.Face:
    wp = cq.Workplane(obj=shape)
    dir_str = NORMAL_MAP[selector.normal]
    faces = wp.faces(dir_str).vals()
    
    idx = selector.index or 0
    if idx >= len(faces): raise ValueError(...)
    
    return faces[idx]

def resolve_edge(shape, selector: EdgeSelector):
    """
    Resolve EdgeSelector to single TopoDS_Edge.
    - normal: projection direction (e.g., "|Z" for vertical edges)
    - length_gt/lt: filter
    - index: pick from filtered
    """
    wp = cq.Workplane(obj=shape)
    
    dir_str = NORMAL_MAP.get(selector.normal)
    if dir_str is None:
        raise ValueError(f"Invalid edge normal: {selector.normal}")
    
    # Base selector: edges in direction
    sel = wp.edges(dir_str)
    
    # Length filter
    if selector.length_gt is not None:
        sel = sel.filter(lambda e: e.Length() > selector.length_gt)
    if selector.length_lt is not None:
        sel = sel.filter(lambda e: e.Length() < selector.length_lt)
    
    edges = sel.vals()
    if not edges:
        raise ValueError(f"No edges match selector: {selector}")
    
    idx = selector.index or 0
    if idx < 0 or idx >= len(edges):
        raise ValueError(f"Edge index {idx} out of range (found {len(edges)})")
    
    return edges[idx]

def resolve_vertex(shape, selector: VertexSelector):
    """
    Resolve to Vector (x,y,z).
    """
    edge = resolve_edge(shape, selector.edge)
    bb = edge.BoundingBox()
    
    if selector.extremum == "min":
        pt = (bb.xmin, bb.ymin, bb.zmin)
    elif selector.extremum == "max":
        pt = (bb.xmax, bb.ymax, bb.zmax)
    elif selector.extremum == "center":
        pt = (bb.xmid, bb.ymid, bb.zmid)
    else:
        raise ValueError(f"Invalid extremum: {selector.extremum}")
    
    return cq.Vector(pt)  # CadQuery Vector

def resolve_face(shape, selector: FaceSelector) -> cq.Face:
    wp = cq.Workplane(obj=shape)
    dir_str = NORMAL_MAP[selector.normal]
    faces = wp.faces(dir_str).vals()
    
    idx = selector.index or 0
    if idx >= len(faces): raise ValueError(...)
    
    return faces[idx]
