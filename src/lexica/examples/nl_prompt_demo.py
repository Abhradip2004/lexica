"""
Lexica IR v1 – Canonical Integration Tests
==========================================

Purpose:
- Validate kernel correctness using IR-only programs
- Generate STEP files for visual inspection
- Lock IR v1 before LLM training

How to run:
    PYTHONPATH=src/lexica python ir_it_tests.py

Each test overwrites the same STEP file.
Inspect after each test.
"""

from lexica.pipeline.orchestration import run_irl_pipeline

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    PrimitiveKind,
    FeatureKind,
    TopologyIntent,
    TopologyTarget,
)


# --------------------------------------------------
# Helper
# --------------------------------------------------

def run_test(name: str, ir_model: IRModel):
    print("\n" + "=" * 60)
    print(f"[TEST] {name}")
    print("=" * 60)

    run_irl_pipeline(ir_model, debug=True)

    print(f"[DONE] {name}")
    print("Inspect generated STEP file before continuing.")
    input("Press ENTER to continue to next test...")


# --------------------------------------------------
# Test 1: Basic Box
# --------------------------------------------------

def test_basic_box():
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 20, "y": 30, "z": 40},
        )
    ])


# --------------------------------------------------
# Test 2: Box + Top Through Hole
# --------------------------------------------------

def test_box_with_through_hole():
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 40, "y": 40, "z": 20},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 8,
                "through_all": True,
                "center": [0, 0],
            },
        )
    ])


# --------------------------------------------------
# Test 3: Box + Chamfer All Edges
# --------------------------------------------------

def test_box_chamfer_all_edges():
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"x": 30, "y": 30, "z": 30},
        ),
        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": 2},
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        )
    ])


# --------------------------------------------------
# Test 4: Simple Shaft (Cylinder)
# --------------------------------------------------

def test_simple_cylinder():
    return IRModel(ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CYLINDER,
            params={"r": 10, "z": 60},
        )
    ])


# --------------------------------------------------
# Test 5: Shaft Collar
# --------------------------------------------------

def test_shaft_collar():
    return IRModel(ops=[
        # Outer cylinder
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CYLINDER,
            params={"r": 20, "z": 20},
        ),

        # Central bore
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 10,
                "through_all": True,
                "center": [0, 0],
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
    ])


# --------------------------------------------------
# Main runner
# --------------------------------------------------

if __name__ == "__main__":
    # run_test("1. Basic Box", test_basic_box())
    # run_test("2. Box with Top Through Hole", test_box_with_through_hole())
    # run_test("3. Box with Chamfered Edges", test_box_chamfer_all_edges())
    # run_test("4. Simple Cylinder (Shaft)", test_simple_cylinder())
    run_test("5. Shaft Collar", test_shaft_collar())

    # print("\n" + "=" * 60)
    # print("ALL TESTS COMPLETED")
    # print("If all STEP files look correct → IR v1 is ready to lock.")
    # print("=" * 60)
