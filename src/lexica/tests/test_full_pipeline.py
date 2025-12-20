from lexica.pipeline.orchestration import run_pipeline


def test_full_pipeline():
    description = """
    Create a box of size 40 by 30 by 20,
    move it 25 units along X,
    rotate it 45 degrees around Z,
    fillet all edges with radius 2.5,
    and export as a STEP file.
    """

    run_pipeline(description)

    print("Full NL --> STEP pipeline test passed")

test_full_pipeline()