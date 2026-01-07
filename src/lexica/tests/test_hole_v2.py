import os

from lexica.pipeline.orchestration import run_irl_pipeline
from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
)
from lexica.pipeline.nl_to_ir.schema import TopologyTarget, TopologyIntent


def test_hole_with_counterbore(tmp_path):
    output_path = tmp_path / "hole_counterbore.step"

    ir = IRModel(
        ops=[
            PrimitiveOp(
                primitive_kind=PrimitiveKind.BOX,
                params={"length": 40, "width": 40, "height": 20},
            ),
            FeatureOp(
                feature_kind=FeatureKind.HOLE,
                params={
                    "diameter": 6,
                    "depth": 15,
                    "face": {"normal": "+Z"},
                    "counterbore": {
                        "diameter": 12,
                        "depth": 4,
                    },
                },
            ),
            ExportOp(
                format="step",
                path=str(output_path),
            ),
        ]
    )

    run_irl_pipeline(ir)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
