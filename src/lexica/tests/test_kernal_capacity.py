from lexica.pipeline.orchestration import run_irl_pipeline
from lexica.pipeline.nl_to_ir.translator import _dict_to_ir
from lexica.pipeline.nl_to_ir.validate import validate_ir


def test_kernel_capacity():
    """
    Kernel capacity stress test.

    Tests:
    - Primitive creation
    - Feature chaining (fillet + chamfer)
    - Multiple tool bodies
    - Transform ops
    - Boolean difference sequencing
    - STEP export

    This bypasses the LLM entirely.
    """

    ir_json = {
        "ops": [
            # --------------------------------------------------
            # Base plate
            # --------------------------------------------------
            {
                "kind": "primitive",
                "primitive_kind": "box",
                "params": {
                    "length": 120,
                    "width": 80,
                    "height": 20
                }
            },

            # --------------------------------------------------
            # Fillet all edges
            # --------------------------------------------------
            # Chamfer first
            {
                "kind": "feature",
                "feature_kind": "chamfer",
                "params": {"distance": 3},
                "topology": {"target": "edge", "rule": "all"}
            },

            # Fillet after
            {
                "kind": "feature",
                "feature_kind": "fillet",
                "params": {"radius": 4},
                "topology": {"target": "edge", "rule": "all"}
            },

            # --------------------------------------------------
            # Center hole
            # --------------------------------------------------
            {
                "kind": "primitive",
                "primitive_kind": "cylinder",
                "params": {
                    "radius": 8,
                    "height": 30
                }
            },
            {
                "kind": "transform",
                "transform_kind": "translate",
                "params": {
                    "dx": 0,
                    "dy": 0,
                    "dz": 0
                }
            },
            {
                "kind": "boolean",
                "boolean_kind": "difference"
            },

            # --------------------------------------------------
            # Corner hole
            # --------------------------------------------------
            {
                "kind": "primitive",
                "primitive_kind": "cylinder",
                "params": {
                    "radius": 4,
                    "height": 30
                }
            },
            {
                "kind": "transform",
                "transform_kind": "translate",
                "params": {
                    "dx": 45,
                    "dy": 25,
                    "dz": 0
                }
            },
            {
                "kind": "boolean",
                "boolean_kind": "difference"
            },

            # --------------------------------------------------
            # Export
            # --------------------------------------------------
            {
                "kind": "export",
                "format": "step",
                "path": "kernel_capacity_test.step"
            }
        ]
    }

    # IR validation
    ir_model = _dict_to_ir(ir_json)
    validate_ir(ir_model)

    # Run IR → IRL → CAD → STEP
    run_irl_pipeline(
        ir_model,
        debug=True
    )

    print("Kernel capacity test completed successfully")


if __name__ == "__main__":
    test_kernel_capacity()
