from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
)
from lexica.examples.run_example import run

ir = IRModel(
    ops=[
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={"length": 100, "width": 60, "height": 10},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 6,
                "through_all": True,
                "face": {"normal": "+Z"},
                "counterbore": {
                    "diameter": 12,
                    "depth": 4,
                },
            },
        ),
        ExportOp(
            format="step",
            path="hole_counterbore.step",
        ),
    ]
)

run(ir)
