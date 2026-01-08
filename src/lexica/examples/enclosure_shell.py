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
            primitive_kind=PrimitiveKind.BOX,
            params={"length": 80, "width": 60, "height": 40},
        ),
        FeatureOp(
            feature_kind=FeatureKind.SHELL,
            params={"thickness": 3},
        ),
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": 5,
                "through_all": True,
                "face": {"normal": "+Z"},
            },
        ),
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": 3},
            topology=TopoPredicate(
                target=TopoTarget.EDGE,
                rule="parallel",
                value="Z",
                index=0,   # 🔑 explicit disambiguation
            ),
        ),
        ExportOp(
            format="step",
            path="enclosure_shell.step",
        ),
    ]
)

run(ir)
