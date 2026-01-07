import cadquery as cq
from lexica.irl.contract import TransformOp


class TransformAdapterError(Exception):
    pass

def _resolve_bbox_pivot(shape, corner: str):
    bb = shape.BoundingBox()

    mapping = {
        "center": (bb.xmid, bb.ymid, bb.zmid),

        "min_x": (bb.xmin, bb.ymid, bb.zmid),
        "max_x": (bb.xmax, bb.ymid, bb.zmid),

        "min_y": (bb.xmid, bb.ymin, bb.zmid),
        "max_y": (bb.xmid, bb.ymax, bb.zmid),

        "min_z": (bb.xmid, bb.ymid, bb.zmin),
        "max_z": (bb.xmid, bb.ymid, bb.zmax),
    }

    if corner not in mapping:
        raise TransformAdapterError(
            f"Invalid bbox pivot corner: {corner}"
        )

    return mapping[corner]


def execute_transform(op: TransformOp, shape):
    params = op.params
    kind = params.get("kind")
    if kind not in ("translate", "rotate"):
        raise TransformAdapterError(f"Invalid transform kind: {kind}")

    wp = cq.Workplane(obj=shape)    

    try:
        if kind == "translate":
            dx = params.get("dx", 0)
            dy = params.get("dy", 0)
            dz = params.get("dz", 0)
            
            if not all(isinstance(v, (int, float)) for v in (dx, dy, dz)):
                raise TransformAdapterError("Translate requires numeric dx, dy, dz")
            
            return wp.translate((dx, dy, dz)).val()

        elif kind == "rotate":
            axis = params.get("axis", "z")
            angle = params.get("angle_deg", 0)
            pivot = params.get("pivot")
            
            if axis not in ("x", "y", "z"):
                raise TransformAdapterError(f"Invalid rotation axis: {axis}")
            
            if not isinstance(angle, (int, float)):
                raise TransformAdapterError("Rotate requires numeric angle_deg")

            axis_vec = {
                "x": (1, 0, 0),
                "y": (0, 1, 0),
                "z": (0, 0, 1),
            }.get(axis)

            if axis_vec is None:
                raise TransformAdapterError(f"Invalid rotation axis: {axis}")

            # wp = cq.Workplane(obj=shape)

            # ---------------------------------
            # Pivot handling
            # ---------------------------------
            if pivot:
                if pivot.get("type") != "bbox":
                    raise TransformAdapterError(
                        f"Unsupported pivot type: {pivot.get('type')}"
                    )

                px, py, pz = _resolve_bbox_pivot(
                    shape, pivot.get("corner", "center")
                )

                # Move pivot to origin
                wp = wp.translate((-px, -py, -pz))

                # Rotate about origin
                wp = wp.rotate(
                    (0, 0, 0),
                    axis_vec,
                    angle,
                )

                # Move back
                wp = wp.translate((px, py, pz))
                return wp.val()

            # ---------------------------------
            # No pivot → rotate about origin
            # ---------------------------------
            return wp.rotate(
                (0, 0, 0),
                axis_vec,
                angle,
            ).val()

        else:
            raise TransformAdapterError(f"Unknown transform kind: {kind}")

    except Exception as e:
        raise TransformAdapterError(f"Transform failed: {e}")


