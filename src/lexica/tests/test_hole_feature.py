import pytest

from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
)
from lexica.cad_engine.executor import IRLExecutor
from lexica.cad_engine.adapters.feature_adapter import FeatureAdapterError


def test_hole_through_all():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 40, "width": 30, "height": 20},
        ),
        FeatureOp(
            op_id="hole",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "hole", "diameter": 6.0, "through_all": True},
        ),
    ]

    executor = IRLExecutor()
    executor.execute(IRLModel(ops=ops))

    assert "body_1" in executor.registry._bodies


def test_hole_with_depth():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 20, "width": 20, "height": 20},
        ),
        FeatureOp(
            op_id="hole",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "hole", "diameter": 5.0, "depth": 10.0},
        ),
    ]

    executor = IRLExecutor()
    executor.execute(IRLModel(ops=ops))

    assert "body_1" in executor.registry._bodies


def test_hole_invalid_diameter_fails():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        FeatureOp(
            op_id="hole",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "hole", "diameter": -2.0, "through_all": True},
        ),
    ]

    with pytest.raises(FeatureAdapterError):
        IRLExecutor().execute(IRLModel(ops=ops))
