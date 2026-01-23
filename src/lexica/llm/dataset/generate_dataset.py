import json
import random
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Tuple

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
    TopologyIntent,
    TopologyTarget,
)

# ============================================================
# Prompt Templates (paraphrases)
# ============================================================

PROMPTS = {
    "mounting_plate_v1": [
        "Create a rectangular mounting plate of {L}mm by {W}mm with thickness {T}mm. Add four corner through holes of {D}mm diameter. Fillet all edges by {R}mm.",
        "Generate a flat plate sized {L} x {W} x {T} millimeters with 4 corner holes (diameter {D}mm). Apply {R}mm fillet to edges.",
        "Design a {T}mm thick plate ({L}mm by {W}mm) with four mounting holes and edge fillets of radius {R}mm.",
        "Make a base plate {L}mm × {W}mm × {T}mm. Drill 4 through holes near the corners ({D}mm). Fillet edges by {R}mm.",
        "I need a mounting plate {L} by {W} mm, thickness {T}mm. Add 4 through holes ({D}mm) and fillet edges {R}mm.",
    ],

    "spacer_v1": [
        "Create a cylindrical spacer with outer diameter {OD}mm and height {H}mm. Add a {D}mm through hole at the center. Chamfer all edges by {C}mm.",
        "Make a standoff: OD {OD}mm, height {H}mm, hole diameter {D}mm through. Add {C}mm chamfer to edges.",
        "Generate a cylinder spacer {OD}mm diameter, {H}mm tall, with a centered through hole {D}mm and {C}mm edge chamfer.",
    ],

    "washer_v1": [
        "Create a washer with outer diameter {OD}mm, inner diameter {ID}mm and thickness {T}mm. Chamfer all edges by {C}mm.",
        "Make a thick ring: OD {OD}mm, ID {ID}mm, thickness {T}mm. Apply {C}mm chamfer to edges.",
        "Generate a washer sized {OD}/{ID}mm with {T}mm thickness and chamfer edges by {C}mm.",
    ],

    "enclosure_v1": [
        "Create a box enclosure {L}mm x {W}mm x {H}mm. Shell it with wall thickness {TH}mm (open top). Export STEP.",
        "Make a rectangular enclosure of {L} by {W} by {H} mm. Apply shell thickness {TH}mm.",
        "Generate a hollow box {L}×{W}×{H}mm with {TH}mm shell thickness.",
    ],

    "drill_test_block_v1": [
        "Create a {S}mm x {S}mm x {T}mm block. Add 3 blind holes on top: {d1}mm depth {z1}mm, {d2}mm depth {z2}mm, {d3}mm depth {z3}mm. Centers spaced {gap}mm.",
        "Make a drilling test block {S}×{S}×{T}mm with three blind holes ({d1}/{d2}/{d3}mm) at depths ({z1}/{z2}/{z3}mm).",
        "Generate a block {S}mm square, thickness {T}mm and drill three blind holes in a line spaced {gap}mm apart.",
    ],

    "plate_center_cutout_v1": [
        "Create a plate {L}mm by {W}mm by {T}mm. Cut a centered through hole of diameter {D}mm. Fillet edges by {R}mm.",
        "Generate a rectangular plate {L}×{W}×{T}mm with a center cutout {D}mm through and {R}mm edge fillet.",
        "Make a flat plate and cut a circular hole {D}mm in the middle. Add {R}mm fillet to edges. Plate size: {L}×{W}×{T}mm.",
    ],

    "cone_with_hole_v1": [
        "Create a frustum cone with bottom radius {r1}mm, top radius {r2}mm, height {H}mm. Add a {D}mm through hole through the center.",
        "Generate a cone frustum: r1={r1}mm, r2={r2}mm, height {H}mm, with a centered through hole {D}mm.",
        "Make a tapered cone (bottom {r1}mm, top {r2}mm, height {H}mm) and drill a {D}mm through hole.",
    ],

    "torus_shell_v1": [
        "Create a torus with major radius {R}mm and minor radius {r}mm. Shell it by {TH}mm.",
        "Generate a torus (R={R}mm, r={r}mm) and apply shell thickness {TH}mm. Export STEP.",
        "Make a hollow torus: major radius {R}mm, minor radius {r}mm with {TH}mm shell thickness.",
    ],
}


# ============================================================
# Helpers
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def choose_prompt(template_id: str, **kwargs) -> str:
    tmpl = random.choice(PROMPTS[template_id])
    return tmpl.format(**kwargs)

def to_dict(obj):
    # Dataclass 
    if is_dataclass(obj):
        return asdict(obj)

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()

    raise TypeError(f"Object of type {type(obj)} is not serializable")


# ============================================================
# IR Builders (canonical structure)
# ============================================================

def ir_mounting_plate(L, W, T, D, margin, fillet_r) -> Dict:
    hx = (L / 2) - margin
    hy = (W / 2) - margin

    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.BOX, params={"x": L, "y": W, "z": T}),

        # Fillet first (more stable)
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": fillet_r},
            topology=TopologyIntent(target=TopologyTarget.EDGE, rule="all"),
        ),

        # 4 corner holes (top face default)
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": D, "through_all": True, "center": [-hx, -hy]}),
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": D, "through_all": True, "center": [hx, -hy]}),
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": D, "through_all": True, "center": [-hx, hy]}),
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": D, "through_all": True, "center": [hx, hy]}),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_spacer(OD, H, hole_d, chamfer_d) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.CYLINDER, params={"r": OD / 2, "z": H}),

        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": hole_d, "through_all": True, "center": [0.0, 0.0]},
        ),

        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": chamfer_d},
            topology=TopologyIntent(target=TopologyTarget.EDGE, rule="all"),
        ),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_washer(OD, ID, T, chamfer_d) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.CYLINDER, params={"r": OD / 2, "z": T}),

        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": ID, "through_all": True, "center": [0.0, 0.0]},
        ),

        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": chamfer_d},
            topology=TopologyIntent(target=TopologyTarget.EDGE, rule="all"),
        ),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_enclosure(L, W, H, TH) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.BOX, params={"x": L, "y": W, "z": H}),
        FeatureOp(feature_kind=FeatureKind.SHELL, params={"thickness": TH}),
        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_drill_test_block(S, T, gap, d1, z1, d2, z2, d3, z3) -> Dict:
    # holes along x-axis
    centers = [-gap, 0, gap]

    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.BOX, params={"x": S, "y": S, "z": T}),

        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": d1, "depth": z1, "center": [centers[0], 0]}),
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": d2, "depth": z2, "center": [centers[1], 0]}),
        FeatureOp(feature_kind=FeatureKind.HOLE, params={"diameter": d3, "depth": z3, "center": [centers[2], 0]}),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_plate_center_cutout(L, W, T, D, fillet_r) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.BOX, params={"x": L, "y": W, "z": T}),

        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": fillet_r},
            topology=TopologyIntent(target=TopologyTarget.EDGE, rule="all"),
        ),

        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": D, "through_all": True, "center": [0.0, 0.0]},
        ),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_cone_with_hole(r1, r2, H, D) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CONE,
            params={"r1": r1, "r2": r2, "z": H},
        ),

        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": D, "through_all": True, "center": [0.0, 0.0]},
        ),

        ExportOp(format="step"),
    ])
    return to_dict(ir)



def ir_torus_shell(R, r, TH) -> Dict:
    ir = IRModel(ops=[
        PrimitiveOp(primitive_kind=PrimitiveKind.TORUS, params={"R": R, "r": r}),
        FeatureOp(feature_kind=FeatureKind.SHELL, params={"thickness": TH}),
        ExportOp(format="step"),
    ])
    return to_dict(ir)



# ============================================================
# Parameter sampling (safe domains)
# ============================================================

def sample_mounting_plate():
    L = random.randint(80, 180)
    W = random.randint(50, 140)
    T = random.randint(4, 10)

    hole_d = random.choice([4, 5, 6, 8])
    margin = random.randint(8, 18)
    margin = max(margin, hole_d)  # safety

    fillet_r = clamp(random.choice([1, 2, 3]), 0.5, T / 2)

    params = dict(L=L, W=W, T=T, D=hole_d, margin=margin, R=fillet_r)
    ir = ir_mounting_plate(L, W, T, hole_d, margin, fillet_r)
    prompt = choose_prompt("mounting_plate_v1", **params, OD=None, H=None, TH=None)

    return prompt, ir, params


def sample_spacer():
    OD = random.choice([16, 18, 20, 22, 24, 26, 30])
    H = random.choice([8, 10, 12, 15, 18, 20])
    hole_d = random.choice([4, 5, 6, 8])
    chamfer_d = random.choice([0.5, 1.0, 1.5])

    # safety
    chamfer_d = min(chamfer_d, (H / 2) - 0.1)

    params = dict(OD=OD, H=H, D=hole_d, C=chamfer_d)
    ir = ir_spacer(OD, H, hole_d, chamfer_d)
    prompt = choose_prompt("spacer_v1", **params)

    return prompt, ir, params


def sample_washer():
    OD = random.choice([30, 35, 40, 45, 50, 60])
    ID = random.choice([10, 12, 15, 18, 20, 25, 30])
    T = random.choice([2, 3, 4, 5, 6])
    chamfer_d = random.choice([0.3, 0.5, 0.8, 1.0])

    # safety: ID must be < OD
    if ID >= OD:
        ID = max(OD - 10, 6)

    params = dict(OD=OD, ID=ID, T=T, C=chamfer_d)
    ir = ir_washer(OD, ID, T, chamfer_d)
    prompt = choose_prompt("washer_v1", **params)

    return prompt, ir, params


def sample_enclosure():
    L = random.randint(60, 180)
    W = random.randint(40, 140)
    H = random.randint(30, 120)
    TH = random.choice([1.5, 2.0, 2.5, 3.0])

    TH = min(TH, min(L, W, H) / 10)

    params = dict(L=L, W=W, H=H, TH=TH)
    ir = ir_enclosure(L, W, H, TH)
    prompt = choose_prompt("enclosure_v1", **params)

    return prompt, ir, params


def sample_drill_test_block():
    S = random.choice([50, 60, 70, 80])
    T = random.choice([15, 18, 20, 25])
    gap = random.choice([12, 15, 18, 20])

    # holes: diameters + depths
    d1 = random.choice([4, 5, 6])
    d2 = random.choice([6, 8])
    d3 = random.choice([8, 10])

    z1 = random.choice([4, 5, 6])
    z2 = random.choice([8, 10, 12])
    z3 = random.choice([12, 14, 15])

    # safety: depth < thickness
    z1 = min(z1, T - 2)
    z2 = min(z2, T - 2)
    z3 = min(z3, T - 2)

    params = dict(S=S, T=T, gap=gap, d1=d1, d2=d2, d3=d3, z1=z1, z2=z2, z3=z3)
    ir = ir_drill_test_block(S, T, gap, d1, z1, d2, z2, d3, z3)
    prompt = choose_prompt("drill_test_block_v1", **params)

    return prompt, ir, params


def sample_plate_center_cutout():
    L = random.randint(90, 220)
    W = random.randint(60, 160)
    T = random.randint(4, 12)
    D = random.choice([20, 25, 30, 40, 50, 60])

    # Safety: cutout diameter must fit in plate
    D = min(D, min(L, W) - 20)

    fillet_r = clamp(random.choice([1, 2, 3]), 0.5, T / 2)

    params = dict(L=L, W=W, T=T, D=D, R=fillet_r)
    ir = ir_plate_center_cutout(L, W, T, D, fillet_r)
    prompt = choose_prompt("plate_center_cutout_v1", **params)

    return prompt, ir, params


def sample_cone_with_hole():
    H = random.randint(30, 120)
    r1 = random.randint(15, 45)
    r2 = random.randint(0, r1 - 5)

    D = random.choice([4, 6, 8, 10])
    D = min(D, (r2 * 2) - 2) if r2 > 6 else min(D, (r1 * 2) - 4)

    D = max(D, 3)

    params = dict(r1=r1, r2=r2, H=H, D=D)
    ir = ir_cone_with_hole(r1, r2, H, D)
    prompt = choose_prompt("cone_with_hole_v1", **params)

    return prompt, ir, params


def sample_torus_shell():
    R = random.choice([25, 30, 35, 40, 45, 50])
    r = random.choice([6, 8, 10, 12, 14])

    # Ensure r < R
    if r >= R:
        r = max(4, R - 5)

    TH = random.choice([1.0, 1.5, 2.0, 2.5])
    TH = min(TH, r / 2 - 0.2)

    params = dict(R=R, r=r, TH=TH)
    ir = ir_torus_shell(R, r, TH)
    prompt = choose_prompt("torus_shell_v1", **params)

    return prompt, ir, params


TEMPLATE_SAMPLERS = [
    ("mounting_plate_v1", sample_mounting_plate),
    ("spacer_v1", sample_spacer),
    ("washer_v1", sample_washer),
    ("enclosure_v1", sample_enclosure),
    ("drill_test_block_v1", sample_drill_test_block),
    ("plate_center_cutout_v1", sample_plate_center_cutout),
    ("cone_with_hole_v1", sample_cone_with_hole),
    ("torus_shell_v1", sample_torus_shell),
]


# ============================================================
# Dataset generator
# ============================================================

def generate_dataset(
    n_samples: int = 2500,
    seed: int = 42,
    out_file: str = "raw_dataset.jsonl",
    balanced: bool = True,
):
    random.seed(seed)

    with open(out_file, "w") as f:
        for i in range(n_samples):
            if balanced:
                tid, sampler = TEMPLATE_SAMPLERS[i % len(TEMPLATE_SAMPLERS)]
            else:
                tid, sampler = random.choice(TEMPLATE_SAMPLERS)

            prompt, ir, params = sampler()

            sample = {
                "template_id": tid,
                "prompt": prompt,
                "params": params,
                "ir": ir,
                "seed": seed,
            }

            f.write(json.dumps(sample) + "\n")

    print(f"Generated {n_samples} samples → {out_file}")
    print(f"balanced={balanced} | templates={len(TEMPLATE_SAMPLERS)}")


if __name__ == "__main__":
    # For CPU training: 2000–3000 
    generate_dataset(n_samples=2400, seed=42, out_file="raw_dataset.jsonl", balanced=True)
