from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
    TopoPredicate,
    TopoTarget,
)
from lexica.cad_engine.executor import IRLExecutor


def test_shell_creates_hollow_body():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 40, "width": 30, "height": 20},
        ),
        FeatureOp(
            op_id="shell",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "shell", "thickness": 2.0},
        ),
    ]

    model = IRLModel(ops=ops)
    executor = IRLExecutor()
    executor.execute(model)

    assert "body_0" not in executor.registry._bodies
    assert "body_1" in executor.registry._bodies
