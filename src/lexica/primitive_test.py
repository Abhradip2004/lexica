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
from lexica.pipeline.nl_to_ir.validate import validate_ir

from lexica.irl.ir_to_irl import lower_ir_to_irl
from lexica.irl.validation import validate_irl
from lexica.cad_engine.executor import IRLExecutor


def run_ir(name: str, ir_model: IRModel):
    """
    Compile IR -> IRL and execute the kernel.
    Output: lexica_output.step (always overwritten).
    """
    print(f"\n=== TEST: {name} ===")

    validate_ir(ir_model)

    irl_model = lower_ir_to_irl(ir_model)
    validate_irl(irl_model)

    ex = IRLExecutor()
    ex.execute(irl_model)

    print("SUCCESS Check: lexica_output.step")


def mounting_plate_ir() -> IRModel:
    """
    Real mechanical part test (MVP kernel):
    - Create a plate
    - Fillet all outer edges (stable)
    - Drill 4 corner holes
    - Export STEP
    """

    # Plate
    L = 120
    W = 80
    T = 6

    # Holes
    hole_d = 6
    margin = 10

    hx = (L / 2) - margin
    hy = (W / 2) - margin

    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": L, "y": W, "z": T},
        ),

        # Fillet first to avoid filleting hole edges later
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": 2},
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        ),

        # Holes (kernel defaults to top face +Z)
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": hole_d, "through_all": True, "center": [-hx, -hy]},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": hole_d, "through_all": True, "center": [hx, -hy]},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": hole_d, "through_all": True, "center": [-hx, hy]},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={"diameter": hole_d, "through_all": True, "center": [hx, hy]},
        ),

        ExportOp(format="step"),
    ])

def spacer_ir() -> IRModel:
    """
    Mechanical sanity test:
    - Cylinder spacer
    - Through hole
    - Chamfer edges
    - Export STEP
    """
    return IRModel(ops=[
        # Outer body
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CYLINDER,
            params={"r": 10, "z": 15},   # OD=20, height=15
        ),

        # Center through hole
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 6,
                "through_all": True,
                "center": [0.0, 0.0]
            },
        ),

        # Chamfer all edges
        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": 1},
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        ),

        ExportOp(format="step"),
    ])

def washer_ir() -> IRModel:
    """
    Mechanical sanity test:
    - Thick washer / ring
    - Through hole (inner diameter)
    - Chamfer edges
    - Export STEP
    """
    return IRModel(ops=[
        # Outer washer body
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CYLINDER,
            params={"r": 20, "z": 4},   # OD=40, thickness=4
        ),

        # Inner through hole (ID=20)
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 20,
                "through_all": True,
                "center": [0.0, 0.0],
            },
        ),

        # Chamfer all edges
        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": 0.5},
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        ),

        ExportOp(format="step"),
    ])

def enclosure_ir() -> IRModel:
    """
    Mechanical sanity test:
    - Enclosure body (box)
    - Shell (hollow) with 2mm thickness
    - Export STEP
    """
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 100, "y": 60, "z": 40},
        ),

        FeatureOp(
            feature_kind=FeatureKind.SHELL,
            params={"thickness": 2},
        ),

        ExportOp(format="step"),
    ])

def drill_test_block_ir() -> IRModel:
    """
    Mechanical sanity test:
    - Create block
    - Add 3 blind holes (depth test)
    - Export STEP
    """
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 60, "y": 60, "z": 20},
        ),

        # Blind hole 1 (left)
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 6,
                "depth": 5,
                "center": [-15, 0],
            },
        ),

        # Blind hole 2 (center)
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 8,
                "depth": 10,
                "center": [0, 0],
            },
        ),

        # Blind hole 3 (right)
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 10,
                "depth": 15,
                "center": [15, 0],
            },
        ),

        ExportOp(format="step"),
    ])

def plate_center_cutout_ir() -> IRModel:
    """
    Mechanical sanity test:
    - Plate
    - Large center cutout (through hole)
    - Fillet edges
    - Export STEP
    """
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 120, "y": 80, "z": 6},
        ),

        # Fillet first (more stable)
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": 2},
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        ),

        # Center cutout
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 50,
                "through_all": True,
                "center": [0.0, 0.0],
            },
        ),

        ExportOp(format="step"),
    ])

def cone_with_hole_ir() -> IRModel:
    """
    Geometry test:
    - Cone / frustum
    - Through hole at center
    - Export STEP
    """
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CONE,
            params={"r1": 30, "r2": 10, "z": 70},
        ),

        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 8,
                "through_all": True,
                "center": [0.0, 0.0],
            },
        ),

        ExportOp(format="step"),
    ])

def torus_shell_ir() -> IRModel:
    """
    Decorative + stress test:
    - Torus primitive
    - Shell (hollow)
    - Export STEP
    """
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.TORUS,
            params={"R": 40, "r": 12},
        ),

        FeatureOp(
            feature_kind=FeatureKind.SHELL,
            params={"thickness": 2},
        ),

        ExportOp(format="step"),
    ])


def main():
    #  MAIN DEMO
    run_ir("torus_shell", torus_shell_ir())

    # --- optional: primitive smoke tests ---
    # run_ir("sphere", IRModel(ops=[
    #     PrimitiveOp(primitive_kind=PrimitiveKind.SPHERE, params={"r": 10}),
    #     ExportOp(format="step"),
    # ]))

    # run_ir("cone", IRModel(ops=[
    #     PrimitiveOp(primitive_kind=PrimitiveKind.CONE, params={"r1": 10, "r2": 0, "z": 25}),
    #     ExportOp(format="step"),
    # ]))

    # run_ir("torus", IRModel(ops=[
    #     PrimitiveOp(primitive_kind=PrimitiveKind.TORUS, params={"R": 20, "r": 5}),
    #     ExportOp(format="step"),
    # ]))


if __name__ == "__main__":
    main()
