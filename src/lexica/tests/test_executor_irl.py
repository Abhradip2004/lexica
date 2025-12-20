from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
    ExportOp,
    TopoPredicate,
    TopoTarget,
)
from lexica.cad_engine.executor import IRLExecutor


def run_test():
    ops = []

    # 1. Create a box
    ops.append(
        PrimitiveOp(
            op_id="box_1",
            reads=[],
            writes="box",
            params={
                "kind": "box",
                "length": 40,
                "width": 30,
                "height": 20,
            },
        )
    )

    # 2. Translate the box
    ops.append(
        FeatureOp(
            op_id="translate_1",
            reads=["box"],
            writes="box_moved",
            params={
                "kind": "translate",
                "dx": 25,
                "dy": 0,
                "dz": 0,
            },
        )
    )

    # 3. Rotate the box
    ops.append(
        FeatureOp(
            op_id="rotate_1",
            reads=["box_moved"],
            writes="box_rotated",
            params={
                "kind": "rotate",
                "axis": "z",
                "angle_deg": 45,
            },
        )
    )

    # 4. Fillet all edges
    ops.append(
        FeatureOp(
            op_id="fillet_1",
            reads=["box_rotated"],
            writes="box_fillet",
            params={
                "kind": "fillet",
                "radius": 2.5,
            },
            topo=TopoPredicate(
                target=TopoTarget.EDGE,
                rule="all",
            ),
        )
    )

    # 5. Export
    ops.append(
        ExportOp(
            op_id="export_1",
            reads=["box_fillet"],
            writes=None,
            params={
                "format": "step",
                "path": "transform_test_output.step",
            },
        )
    )

    model = IRLModel(ops=ops)

    executor = IRLExecutor()
    executor.execute(model)

    print("Transform + fillet IRL test completed successfully")


if __name__ == "__main__":
    run_test()
