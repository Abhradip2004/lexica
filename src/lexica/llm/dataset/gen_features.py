import json
import random
from pathlib import Path


# -------------------------------------------------
# Config
# -------------------------------------------------

OUTPUT_FILE = Path(__file__).parent / "features.jsonl"
NUM_SAMPLES = 2000

SYSTEM_PROMPT = (
    Path(__file__).parents[1] / "prompts" / "system.txt"
).read_text().strip()


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _rand_int(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _export_step_op() -> dict:
    # IR schema uses ExportOp(format="step")
    return {"kind": "export", "format": "step", "params": {}}


def _wrap_sample(user_text: str, ir: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(ir)},
        ]
    }


def _topo_all_edges() -> dict:
    # Topology intent: apply to all edges
    return {"target": "edge", "rule": "all"}


# -------------------------------------------------
# Primitive builders
# -------------------------------------------------

def prim_box():
    x, y, z = _rand_int(30, 150), _rand_int(30, 150), _rand_int(20, 120)
    text = f"make a box of size {x} {y} {z}"
    op = {"kind": "primitive", "primitive_kind": "box", "params": {"x": x, "y": y, "z": z}}
    return text, op, min(x, y, z)


def prim_cylinder():
    r = _rand_int(10, 70)
    z = _rand_int(30, 160)
    text = f"make a cylinder of radius {r} and height {z}"
    op = {"kind": "primitive", "primitive_kind": "cylinder", "params": {"r": r, "z": z}}
    return text, op, min(r, z)


def prim_sphere():
    r = _rand_int(15, 90)
    text = f"make a sphere of radius {r}"
    op = {"kind": "primitive", "primitive_kind": "sphere", "params": {"r": r}}
    return text, op, r


def prim_cone():
    r1 = _rand_int(15, 80)
    z = _rand_int(40, 170)

    # Sometimes frustum
    if random.random() < 0.7:
        r2 = 0
        text = f"make a cone of base radius {r1} and height {z}"
    else:
        r2 = _rand_int(3, max(4, r1 - 1))
        text = f"make a frustum with base radius {r1}, top radius {r2}, and height {z}"

    op = {"kind": "primitive", "primitive_kind": "cone", "params": {"r1": r1, "r2": r2, "z": z}}
    return text, op, min(r1, z)


def prim_torus():
    R = _rand_int(25, 140)
    r = _rand_int(5, max(6, int(R * 0.40)))
    if r >= R:
        r = max(1, R - 1)

    text = f"make a torus with major radius {R} and minor radius {r}"
    op = {"kind": "primitive", "primitive_kind": "torus", "params": {"R": R, "r": r}}
    return text, op, r


PRIMITIVE_GENERATORS = [
    ("box", prim_box),
    ("cylinder", prim_cylinder),
    ("sphere", prim_sphere),
    ("cone", prim_cone),
    ("torus", prim_torus),
]


# -------------------------------------------------
# Feature builders (schema-correct)
# -------------------------------------------------

def feat_fillet(max_dim: int):
    radius = _rand_int(1, max(2, max_dim // 8))
    text = f"fillet all edges with radius {radius}"
    op = {
        "kind": "feature",
        "feature_kind": "fillet",
        "params": {"radius": radius},
        "topology": _topo_all_edges(),
    }
    return text, op


def feat_chamfer(max_dim: int):
    distance = _rand_int(1, max(2, max_dim // 10))
    text = f"chamfer all edges by distance {distance}"
    op = {
        "kind": "feature",
        "feature_kind": "chamfer",
        "params": {"distance": distance},
        "topology": _topo_all_edges(),
    }
    return text, op


def feat_shell(max_dim: int):
    thickness = _rand_int(1, max(2, max_dim // 12))
    text = f"shell the body with thickness {thickness}"
    op = {
        "kind": "feature",
        "feature_kind": "shell",
        "params": {"thickness": thickness},
    }
    return text, op


def feat_hole(max_dim: int):
    diameter = _rand_int(5, max(6, max_dim // 2))

    # Through hole is safest for diverse primitives
    if random.random() < 0.7:
        text = f"make a centered through hole of diameter {diameter}"
        op = {
            "kind": "feature",
            "feature_kind": "hole",
            "params": {
                "diameter": diameter,
                "through_all": True,
                "center": [0.0, 0.0],
            },
        }
    else:
        depth = _rand_int(5, max(6, max_dim))
        text = f"make a centered blind hole of diameter {diameter} and depth {depth}"
        op = {
            "kind": "feature",
            "feature_kind": "hole",
            "params": {
                "diameter": diameter,
                "depth": depth,
                "through_all": False,
                "center": [0.0, 0.0],
            },
        }

    return text, op


# -------------------------------------------------
# Feature validity per primitive (kernel-safe)
# -------------------------------------------------

# Fillet/chamfer require edges.
# Sphere has no edges unless modified. Torus also usually has no "sharp edges".
_ALLOWED_FEATURES_BY_PRIM = {
    "box": ["fillet", "chamfer", "shell", "hole"],
    "cylinder": ["fillet", "chamfer", "shell", "hole"],
    "cone": ["fillet", "chamfer", "shell", "hole"],
    "sphere": ["shell", "hole"],
    "torus": ["shell", "hole"],
}

_FEATURE_FACTORIES = {
    "fillet": feat_fillet,
    "chamfer": feat_chamfer,
    "shell": feat_shell,
    "hole": feat_hole,
}


# -------------------------------------------------
# Dataset generator
# -------------------------------------------------

def gen_sample():
    prim_name, prim_fn = random.choice(PRIMITIVE_GENERATORS)
    prim_text, prim_op, scale = prim_fn()

    allowed = _ALLOWED_FEATURES_BY_PRIM[prim_name]
    feat_kind = random.choice(allowed)

    feat_text, feat_op = _FEATURE_FACTORIES[feat_kind](scale)

    user_text = f"{prim_text}. Then {feat_text}."

    ir = {
        "ops": [
            prim_op,
            feat_op,
            _export_step_op(),
        ]
    }

    return user_text, ir


def build_dataset(n: int):
    samples = []
    for _ in range(n):
        text, ir = gen_sample()
        samples.append(_wrap_sample(text, ir))
    return samples


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    data = build_dataset(NUM_SAMPLES)

    with OUTPUT_FILE.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(data)} samples to {OUTPUT_FILE}")
