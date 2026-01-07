import cadquery as cq
from lexica.irl.contract import FaceSelector

def resolve_face(shape, selector: FaceSelector):
    """
    Resolve FaceSelector to single TopoDS_Face.
    
    Semantics:
    - normal -> CadQuery direction filter
    - index picks from filtered list (0-first)
    - Raises if index out-of-bounds or no faces
    """
    wp = cq.Workplane(obj=shape)
    
    # Full normal -> CadQuery direction map
    normal_map = {
        "+X": ">X",
        "-X": "<X",
        "+Y": ">Y",
        "-Y": "<Y",
        "+Z": ">Z",
        "-Z": "<Z",
    }
    
    dir_str = normal_map.get(selector.normal)
    if dir_str is None:
        raise ValueError(f"Invalid face normal: {selector.normal}")
    
    # Filter faces by direction
    faces = getattr(wp, dir_str)().vals()
    
    if not faces:
        raise ValueError(f"No faces found for normal '{selector.normal}'")
    
    if selector.index is None:
        if len(faces) != 1:
            raise ValueError(f"Ambiguous faces for '{selector.normal}' (found {len(faces)}, need index)")
        return faces[0]
    
    if selector.index < 0 or selector.index >= len(faces):
        raise ValueError(
            f"Face index {selector.index} out of range "
            f"(normal '{selector.normal}' found {len(faces)} faces)"
        )
    
    return faces[selector.index]  # single TopoDS_Face
