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
            params={"length": 40, "width": 30, "height": 20},
        ),
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={"radius": 2},
            topology=TopoPredicate(
                target=TopoTarget.EDGE,
                rule="all",   # legacy compatibility
            ),
        ),
        ExportOp(
            format="step",
            path="fillet_topology_v2.step",
        ),
    ]
)

run(ir)
