"""
L-Bracket Example (Canonical, Correct)
=====================================

Generates a true L-shaped mounting bracket using explicit
kernel-grade operations:

- Two rectangular plates
- One plate rotated 90 degrees
- Explicit spatial translation
- Boolean union
- Filleted edges
- Mounting holes
- STEP export

This file is intentionally verbose and explicit to demonstrate
correct CAD kernel semantics.
"""

from lexica.pipeline.orchestration import run_irl_pipeline
from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    PrimitiveOp,
    TransformOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    PrimitiveKind,
    TransformKind,
    FeatureKind,
    BooleanKind,
    TopologyIntent,
    TopologyTarget,
)

# --------------------------------------------------
# Geometry parameters (mm)
# --------------------------------------------------
PLATE_LENGTH = 60
PLATE_HEIGHT = 80
PLATE_THICKNESS = 10
FILLET_RADIUS = 3
HOLE_DIAMETER = 8

# --------------------------------------------------
# IR Program
# --------------------------------------------------

ir_model = IRModel(
    ops=[
        # --------------------------------------------------
        # Vertical plate (stands upright)
        # --------------------------------------------------
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={
                "length": PLATE_LENGTH,
                "width": PLATE_THICKNESS,
                "height": PLATE_HEIGHT,
            },
        ),

        # --------------------------------------------------
        # Horizontal plate (initially flat)
        # --------------------------------------------------
        PrimitiveOp(
            primitive_kind=PrimitiveKind.BOX,
            params={
                "length": PLATE_LENGTH,
                "width": PLATE_HEIGHT,
                "height": PLATE_THICKNESS,
            },
        ),

        # --------------------------------------------------
        # Rotate horizontal plate 90° about X-axis
        # (so it becomes vertical)
        # --------------------------------------------------
        TransformOp(
            transform_kind=TransformKind.ROTATE,
            params={
                "axis": "x",
                "angle_deg": 90,
            },
        ),

        # --------------------------------------------------
        # Translate rotated plate upward so it meets the
        # first plate at a right angle
        # --------------------------------------------------
        TransformOp(
            transform_kind=TransformKind.TRANSLATE,
            params={
                "dx": 0,
                "dy": 0,
                "dz": (PLATE_HEIGHT / 2) - (PLATE_THICKNESS / 2),
            },
        ),

        # --------------------------------------------------
        # Boolean union → true L-bracket
        # --------------------------------------------------
        BooleanOp(
            boolean_kind=BooleanKind.UNION,
            operands=[0, 3],
        ),

        # --------------------------------------------------
        # Fillet all edges
        # --------------------------------------------------
        FeatureOp(
            feature_kind=FeatureKind.FILLET,
            params={
                "radius": FILLET_RADIUS,
            },
            topology=TopologyIntent(
                target=TopologyTarget.EDGE,
                rule="all",
            ),
        ),

        # --------------------------------------------------
        # Mounting hole on vertical plate
        # --------------------------------------------------
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": HOLE_DIAMETER,
                "through_all": True,
                "face": {
                    "normal": "+X",
                    "index": 0,
                },
                "center": (0, 20),
            },
        ),

        # --------------------------------------------------
        # Mounting hole on horizontal plate
        # --------------------------------------------------
        FeatureOp(
            feature_kind=FeatureKind.HOLE,
            params={
                "diameter": HOLE_DIAMETER,
                "through_all": True,
                "face": {
                    "normal": "+Z",
                    "index": 0,
                },
                "center": (0, 20),
            },
        ),

        # --------------------------------------------------
        # Export STEP
        # --------------------------------------------------
        ExportOp(
            format="step",
            path="l_bracket.step",
        ),
    ]
)

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

if __name__ == "__main__":
    run_irl_pipeline(ir_model, debug=True)
