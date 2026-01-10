import random
from pathlib import Path
import json

# -------------------------------------------------
# Config
# -------------------------------------------------

OUTPUT_FILE = Path("box_primitives.jsonl")
NUM_SAMPLES = 500  

# Load system prompt (single source of truth)
SYSTEM_PROMPT = (
    Path(__file__)
    .parents[1]
    /"prompts"
    / "system.txt"
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
    "generate a rectangle with measurements {x} {y} {z}",
    "design a rectangular box with {x} length {y} breadth {z} height"
    "build a rectangular shape with dimensions width {x} length {y} height {z}"
]

CUBE_TEMPLATES = [
    "make a cube of side {s}",
    "create a cube {s}",
    "perfect cube with side {s}",
    "equal box {s} by {s} by {s}",
    "cube with size {s}",
    "make a {s} unit cube",
]


# -------------------------------------------------
# Sample generators
# -------------------------------------------------

def gen_box_sample():
    x = random.randint(5, 100)
    y = random.randint(5, 100)
    z = random.randint(5, 100)

    text = random.choice(BOX_TEMPLATES).format(x=x, y=y, z=z)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "box",
                "params": {"x": x, "y": y, "z": z},
            }
        ]
    }

    return text, ir


def gen_cube_sample():
    s = random.randint(5, 100)

    text = random.choice(CUBE_TEMPLATES).format(s=s)

    ir = {
        "ops": [
            {
                "kind": "primitive",
                "primitive_kind": "box",
                "params": {"x": s, "y": s, "z": s},
            }
        ]
    }

    return text, ir


# -------------------------------------------------
# Dataset builder
# -------------------------------------------------

def build_dataset(n: int):
    samples = []

    for _ in range(n):
        if random.random() < 0.6:
            text, ir = gen_box_sample()
        else:
            text, ir = gen_cube_sample()

        samples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": json.dumps(ir)},
                ]
            }
        )

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
