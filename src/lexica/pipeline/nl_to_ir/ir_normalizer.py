from typing import Dict, Any


from typing import Dict, Any


def normalize_ir(data: Dict[str, Any]) -> Dict[str, Any]:
    # -------------------------------------------------
    # 1. Unwrap common root wrappers
    # -------------------------------------------------

    for wrapper in ("object", "model", "shape", "geometry"):
        if isinstance(data, dict) and wrapper in data and isinstance(data[wrapper], dict):
            data = data[wrapper]
            break

    # -------------------------------------------------
    # 2. Normalize ops container
    # -------------------------------------------------

    if "operation" in data and "ops" not in data:
        data["ops"] = [data.pop("operation")]

    if "operations" in data and "ops" not in data:
        data["ops"] = data.pop("operations")

    if "ops" in data and isinstance(data["ops"], dict):
        data["ops"] = [data["ops"]]

    ops = data.get("ops", [])

    # -------------------------------------------------
    # 3. ROOT-LEVEL GEOMETRY FIX (THIS WAS MISSING)
    # -------------------------------------------------

    if "size" in data and ops:
        first_op = ops[0]
        params = first_op.setdefault("params", {})
        params["size"] = data.pop("size")

    if "dimensions" in data and ops:
        first_op = ops[0]
        params = first_op.setdefault("params", {})
        params["dimensions"] = data.pop("dimensions")

    # -------------------------------------------------
    # 4. Per-operation normalization
    # -------------------------------------------------

    for op in ops:
        if not isinstance(op, dict):
            continue

        # type -> kind
        if "type" in op and "kind" not in op:
            op["kind"] = op.pop("type")

        params = op.setdefault("params", {})

        # op-level size -> params
        if "size" in op:
            params["size"] = op.pop("size")

        if "dimensions" in op:
            params["dimensions"] = op.pop("dimensions")

        # dimensions -> x,y,z
        if "dimensions" in params:
            dims = params.pop("dimensions")
            if _is_xyz_triplet(dims):
                params["x"], params["y"], params["z"] = dims

        # size -> x,y,z
        if "size" in params:
            size = params.pop("size")
            if _is_xyz_triplet(size):
                params["x"], params["y"], params["z"] = size

        # short aliases
        if "r" in params and "radius" not in params:
            params["radius"] = params.pop("r")

        if "d" in params and "distance" not in params:
            params["distance"] = params.pop("d")

        op["params"] = params

    # -------------------------------------------------
    # 5. FINAL ROOT CLEANUP (ABSOLUTELY CRITICAL)
    # -------------------------------------------------

    # IRModel only accepts 'ops'
    data = {k: v for k, v in data.items() if k == "ops"}

    return data



# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _is_xyz_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )



def _is_xyz_triplet(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )
