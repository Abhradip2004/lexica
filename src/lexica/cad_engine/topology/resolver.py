import cadquery as cq
from lexica.irl.contract import FaceSelector


def resolve_face(shape, selector: FaceSelector):
    wp = cq.Workplane(obj=shape)

    normal_map = {
        "+X": ">X",
        "-X": "<X",
        "+Y": ">Y",
        "-Y": "<Y",
        "+Z": ">Z",
        "-Z": "<Z",
    }

    cq_sel = normal_map[selector.normal]

    if selector.index is None:
        return cq_sel  # selector string ONLY

    faces = wp.faces(cq_sel).vals()

    if selector.index >= len(faces):
        raise ValueError(
            f"Face index {selector.index} out of range "
            f"(found {len(faces)} faces)"
        )

    return faces[selector.index]  # single TopoDS_Face
