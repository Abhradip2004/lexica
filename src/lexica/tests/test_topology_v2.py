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
from lexica.irl.contract import TopoPredicate, TopoTarget


def test_face_max_z_selection(tmp_path):
    output_path = tmp_path / "face_max_z.step"

    ir = IRModel(
        ops=[
            PrimitiveOp(
                primitive_kind=PrimitiveKind.BOX,
                params={"length": 20, "width": 20, "height": 20},
            ),
            FeatureOp(
                feature_kind=FeatureKind.HOLE,
                params={
                    "diameter": 4,
                    "through_all": True,
                    "face": {"normal": "+Z"},  # hole still uses face selector
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
