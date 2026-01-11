import random
import json
from pathlib import Path

# -------------------------------------------------
# Config
# -------------------------------------------------

OUTPUT_FILE = Path("feature_shapes.jsonl")
NUM_SAMPLES = 600   # good starting point

SYSTEM_PROMPT = (
    Path(__file__)
    .parents[1]
    /"prompts"
    / "system.txt"
).read_text().strip()

# -------------------------------------------------
# Language templates
# -------------------------------------------------

FILLET_TEMPLATES = [
    "make a box {x} by {y} by {z} and round all edges by {r}",
    "create a {x}x{y}x{z} box and fillet the edges with radius {r}",
    "box of size {x} {y} {z} with rounded edges of {r}",
]

CHAMFER_TEMPLATES = [
    "make a box {x} {y} {z} and chamfer all edges by {d}",
    "create a box of {x}x{y}x{z} with edge chamfer {d}",
    "box {x} by {y} by {z} with chamfered edges {d}",
]

HOLE_TEMPLATES = [
    "make a box {x} {y} {z} and drill a hole of diameter {d} in the center",
    "create a {x}x{y}x{z} block with a central hole of {d}",
    "box of size {x} {y} {z} with a through hole {d}",
]

# -------------------------------------------------
# IR builders
# -------------------------------------------------

def base_box(x, y, z):
    return {
        "kind": "primitive",
        "primitive_kind": "box",
        "params": {"x": x, "y": y, "z": z},
    }


def fillet_feature(r):
    return {
        "kind": "feature",
        "feature_kind": "fillet",
        "params": {"radius": r},
        "topology": {"target": "edge", "rule": "all"},
    }


def chamfer_feature(d):
    return {
        "kind": "feature",
        "feature_kind": "chamfer",
        "params": {"distance": d},
        "topology": {"target": "edge", "rule": "all"},
    }


def hole_feature(d):
    return {
        "kind": "feature",
        "feature_kind": "hole",
        "params": {
            "diameter": d,
            "through_all": True,
            "center": [0, 0],
        },
    }

# -------------------------------------------------
# Sample generators
# -------------------------------------------------

def gen_fillet_shape():
    x, y, z = [random.randint(20, 100) for _ in range(3)]
    r = random.randint(1, min(x, y, z) // 5)

    text = random.choice(FILLET_TEMPLATES).format(x=x, y=y, z=z, r=r)

    return text, {
        "ops": [
            base_box(x, y, z),
            fillet_feature(r),
        ]
    }


def gen_chamfer_shape():
    x, y, z = [random.randint(20, 100) for _ in range(3)]
    d = random.randint(1, min(x, y, z) // 6)

    text = random.choice(CHAMFER_TEMPLATES).format(x=x, y=y, z=z, d=d)

    return text, {
        "ops": [
            base_box(x, y, z),
            chamfer_feature(d),
        ]
    }


def gen_hole_shape():
    x, y, z = [random.randint(30, 120) for _ in range(3)]
    d = random.randint(5, min(x, y) // 3)

    text = random.choice(HOLE_TEMPLATES).format(x=x, y=y, z=z, d=d)

    return text, {
        "ops": [
            base_box(x, y, z),
            hole_feature(d),
        ]
    }

# -------------------------------------------------
# Dataset builder
# -------------------------------------------------

def build_dataset(n):
    samples = []

    for _ in range(n):
        p = random.random()
        if p < 0.33:
            text, ir = gen_fillet_shape()
        elif p < 0.66:
            text, ir = gen_chamfer_shape()
        else:
            text, ir = gen_hole_shape()

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(ir)},
            ]
        })

    return samples

# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    data = build_dataset(NUM_SAMPLES)

    with OUTPUT_FILE.open("w") as f:
        for s in data:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(data)} feature shape samples to {OUTPUT_FILE}")
