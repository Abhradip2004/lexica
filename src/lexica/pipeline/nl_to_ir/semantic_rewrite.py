from __future__ import annotations


def rewrite_semantic_aliases_dict(data: dict) -> dict:
    """
    Rewrite semantic aliases on raw IR JSON dict
    BEFORE enum conversion.
    """

    if "ops" not in data:
        return data

    for op in data["ops"]:
        # cube -> box
        if (
            op.get("kind") == "primitive"
            and op.get("primitive_kind") == "cube"
        ):
            side = op.get("params", {}).get("side")
            if side is None:
                raise ValueError("Cube primitive requires 'side' param")

            op["primitive_kind"] = "box"
            op["params"] = {
                "x": side,
                "y": side,
                "z": side,
            }

    return data
