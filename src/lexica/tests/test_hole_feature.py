import pytest

from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
)

from lexica.cad_engine.executor import IRLExecutor
from lexica.cad_engine.adapters.feature_adapter import FeatureAdapterError


def test_hole_with_counterbore():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 40, "width": 40, "height": 20},
        ),
        FeatureOp(
            op_id="hole_cb",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={
                "kind": "hole",
                "diameter": 6.0,
                "through_all": True,
                "counterbore": {
                    "diameter": 12.0,
                    "depth": 4.0,
                },
            },
        ),
    ]

    IRLExecutor().execute(IRLModel(ops=ops))


def test_hole_with_countersink():
    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 40, "width": 40, "height": 20},
        ),
        FeatureOp(
            op_id="hole_cs",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={
                "kind": "hole",
                "diameter": 6.0,
                "through_all": True,
                "countersink": {
                    "diameter": 12.0,
                    "angle_deg": 90,
                },
            },
        ),
    ]

    IRLExecutor().execute(IRLModel(ops=ops))
