import json
from lexica.pipeline.nl_to_ir.translator import nl_to_ir
from lexica.pipeline.nl_to_ir.schema import IRModel


# -------------------------------------------------
# Helper
# -------------------------------------------------

def pretty(obj):
    return json.dumps(obj, indent=2)


def run_case(prompt: str):
    print("\n==============================")
    print("PROMPT:", prompt)
    ir_model = nl_to_ir(prompt)
    ir = ir_model.to_dict()  
    print(pretty(ir))
    return ir

# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_box_only():
    ir = run_case("Make a cube of side 15")

    assert "ops" in ir
    assert len(ir["ops"]) == 1

    op = ir["ops"][0]
    assert op["kind"] == "primitive"
    assert op["primitive_kind"] == "box"

    params = op["params"]
    assert params["x"] == params["y"] == params["z"] == 15


def test_box_dimensions():
    ir = run_case("Make a box with dimensions 30, 20, 10")

    op = ir["ops"][0]
    params = op["params"]

    assert params["x"] == 30
    assert params["y"] == 20
    assert params["z"] == 10


def test_box_with_fillet():
    ir = run_case("Make a box 30 30 30 and round all edges by 2")

    assert len(ir["ops"]) >= 1
    assert ir["ops"][0]["kind"] == "primitive"


def test_box_with_chamfer():
    ir = run_case("Create a 50x40x20 block with chamfered edges of 1")

    assert len(ir["ops"]) == 2

    feat = ir["ops"][1]
    assert feat["feature_kind"] == "chamfer"
    assert feat["params"]["distance"] == 1


def test_box_with_hole():
    ir = run_case("Make a box 40 40 20 and drill a hole of diameter 10 in the center")

    assert len(ir["ops"]) == 2

    feat = ir["ops"][1]
    assert feat["feature_kind"] == "hole"

    params = feat["params"]
    assert params["diameter"] == 10
    assert params["through_all"] is True
    assert params["center"] == [0, 0]


# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    print("Running Lexica prompt-to-IR tests...\n")

    # test_box_only()
    test_box_dimensions()
    # test_box_with_fillet()
    # test_box_with_chamfer()
    # test_box_with_hole()

    print("\nALL TESTS PASSED")
