import json
import random
from pathlib import Path


# -------------------------------------------------
# Config
# -------------------------------------------------

OUTPUT_FILE = Path(__file__).parent / "primitives.jsonl"
NUM_SAMPLES = 2000

# Load system prompt (single source of truth)
SYSTEM_PROMPT = (
    Path(__file__).parents[1] / "prompts" / "system.txt"
).read_text().strip()


# -------------------------------------------------
# Language templates
# -------------------------------------------------

BOX_TEMPLATES = [
    "make a box of size {x} {y} {z}",
    "create a box {x} by {y} by {z}",
    "box with dimensions {x}, {y}, {z}",
    "make a rectangular box {x} x {y} x {z}",
    "create a block of {x} {y} {z}",
    "design a box with sides {x} {y} {z}",
    "create a solid box with dimensions {x} {y} {z}",
    "design a rectangular box with length {x} breadth {y} height {z}",
    "build a rectangular solid of width {x} length {y} height {z}",
]

CUBE_TEMPLATES = [
    "make a cube of side {s}",
    "create a cube {s}",
    "perfect cube with side {s}",
    "equal box {s} by {s} by {s}",
    "cube with size {s}",
    "make a {s} unit cube",
]

CYLINDER_TEMPLATES = [
    "make a cylinder of radius {r} and height {z}",
    "create a cylinder r={r} height={z}",
    "cylinder with r {r} and height {z}",
    "generate a cylindrical solid of radius {r} and height {z}",
    "design a cylinder with radius {r} and height {z}",
]

SPHERE_TEMPLATES = [
    "make a sphere of radius {r}",
    "create a sphere r={r}",
    "generate a spherical ball of radius {r}",
    "design a sphere with radius {r}",
    "create a solid sphere radius {r}",
]

CONE_TEMPLATES = [
    "make a cone of base radius {r1} and height {z}",
    "create a cone radius {r1} height {z}",
    "generate a cone with radius {r1} and height {z}",
    "design a cone with base radius {r1} height {z}",
]

FRUSTUM_TEMPLATES = [
    "make a conical frustum with base radius {r1} top radius {r2} and height {z}",
    "create a frustum r1={r1} r2={r2} height={z}",
    "design a truncated cone with base radius {r1} top radius {r2} height {z}",
]

TORUS_TEMPLATES = [
    "make a torus with major radius {R} and minor radius {r}",
    "create a torus R={R} r={r}",
    "generate a donut with major radius {R} and tube radius {r}",
    "design a torus with R {R} and r {r}",
]


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _rand_dim(lo: int = 5, hi: int = 100) -> int:
    return random.randint(lo, hi)


def _export_step_op() -> dict:
    # Your export op in IR uses "format" (not export_kind enum).
    return {"kind": "export", "format": "step", "params": {}}


def _wrap_sample(user_text: str, ir: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(ir)},
        ]
    }


# -------------------------------------------------
# Sample generators
# -------------------------------------------------

def gen_box_sample():
    x, y, z = _rand_dim(), _rand_dim(), _rand_dim()
    text = random.choice(BOX_TEMPLATES).format(x=x, y=y, z=z)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "box",
                "params": {"x": x, "y": y, "z": z},
            },
            _export_step_op(),
        ]
    }
    return text, ir


def gen_cube_sample():
    s = _rand_dim()
    text = random.choice(CUBE_TEMPLATES).format(s=s)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "box",
                "params": {"x": s, "y": s, "z": s},
            },
            _export_step_op(),
        ]
    }
    return text, ir


def gen_cylinder_sample():
    r = _rand_dim(3, 60)
    z = _rand_dim(5, 150)
    text = random.choice(CYLINDER_TEMPLATES).format(r=r, z=z)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "cylinder",
                "params": {"r": r, "z": z},
            },
            _export_step_op(),
        ]
    }
    return text, ir


def gen_sphere_sample():
    r = _rand_dim(3, 80)
    text = random.choice(SPHERE_TEMPLATES).format(r=r)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "sphere",
                "params": {"r": r},
            },
            _export_step_op(),
        ]
    }
    return text, ir


def gen_cone_sample():
    """
    Generates both true cones (r2=0) and frustums.
    Kernel expects cone params: r1, r2, z.
    """
    z = _rand_dim(10, 200)
    r1 = _rand_dim(5, 100)

    if random.random() < 0.65:
        # cone
        r2 = 0
        text = random.choice(CONE_TEMPLATES).format(r1=r1, z=z)
    else:
        # frustum (ensure r2 < r1)
        r2 = random.randint(1, max(1, r1 - 1))
        text = random.choice(FRUSTUM_TEMPLATES).format(r1=r1, r2=r2, z=z)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "cone",
                "params": {"r1": r1, "r2": r2, "z": z},
            },
            _export_step_op(),
        ]
    }
    return text, ir


def gen_torus_sample():
    """
    Kernel expects torus params: R (major) and r (minor), with r < R.
    """
    R = _rand_dim(10, 150)
    r = random.randint(2, max(2, int(R * 0.45)))  # keep safely < R
    if r >= R:
        r = max(1, R - 1)

    text = random.choice(TORUS_TEMPLATES).format(R=R, r=r)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "torus",
                "params": {"R": R, "r": r},
            },
            _export_step_op(),
        ]
    }
    return text, ir


# -------------------------------------------------
# Dataset builder
# -------------------------------------------------

def build_dataset(n: int):
    """
    Build a dataset of primitive-only tasks.

    Sampling strategy intentionally mixes primitives rather than heavily
    overfitting to boxes/cubes.
    """
    generators = [
        ("box", gen_box_sample, 0.20),
        ("cube", gen_cube_sample, 0.15),
        ("cylinder", gen_cylinder_sample, 0.20),
        ("sphere", gen_sphere_sample, 0.15),
        ("cone", gen_cone_sample, 0.15),
        ("torus", gen_torus_sample, 0.15),
    ]

    # Build a weighted choice list once
    choices = []
    for name, fn, w in generators:
        choices.append((fn, w))

    samples = []
    for _ in range(n):
        fn = _weighted_choice(choices)
        text, ir = fn()
        samples.append(_wrap_sample(text, ir))

    return samples


def _weighted_choice(items):
    """
    items: list of (value, weight)
    """
    r = random.random()
    s = 0.0
    for val, w in items:
        s += w
        if r <= s:
            return val
    return items[-1][0]


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    data = build_dataset(NUM_SAMPLES)

    with OUTPUT_FILE.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(data)} samples to {OUTPUT_FILE}")
