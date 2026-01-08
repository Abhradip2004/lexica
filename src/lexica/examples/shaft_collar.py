from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
)
from lexica.irl.contract import TopoPredicate, TopoTarget
from lexica.examples.run_example import run

ir = IRModel(
    ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.CYLINDER,
            params={
                "radius": 20,   # ✅ correct
                "height": 20,
            },
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 20,
                "through_all": True,
                "face": {"normal": "+Z"},
            },
        ),
        FeatureOp(
            feature_kind=FeatureKind.CHAMFER,
            params={"distance": 1},
            topology=TopoPredicate(
                target=TopoTarget.FACE,
                rule="max",
                value="Z",
                # index=0,
            ),
        ),
        ExportOp(
            format="step",
            path="shaft_collar.step",
        ),
    ]
)

run(ir)
