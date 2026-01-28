"""
Torque CLI

IR -> CAD compiler interface.

This CLI executes Torque standalone, without Orbit.
"""

import json
import argparse
from pathlib import Path

from lexica.torque.language.ir.schema import (
    IRModel,
    IROp,
    IROpKind,
)
from lexica.torque.language.ir.validate import validate_ir
from lexica.torque.irl.ir_to_irl import lower_ir_to_irl
from lexica.torque.pipeline.run_irl import run_irl

from lexica.torque.language.ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    TransformOp,
    BooleanOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
    TransformKind,
    BooleanKind,
)


class TorqueCLIError(Exception):
    """User-facing Torque CLI error."""
    pass



def load_ir(path: Path) -> IRModel:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        raise TorqueCLIError(f"Failed to read IR file: {e}") from e

    ops = []

    for idx, raw in enumerate(data.get("ops", [])):
        try:
            kind = raw["kind"]

            if kind == "primitive":
                op = PrimitiveOp(
                    primitive_kind=PrimitiveKind(raw["primitive_kind"]),
                    params=raw.get("params", {}),
                )

            elif kind == "transform":
                op = TransformOp(
                    transform_kind=TransformKind(raw["transform_kind"]),
                    params=raw.get("params", {}),
                )

            elif kind == "feature":
                op = FeatureOp(
                    feature_kind=FeatureKind(raw["feature_kind"]),
                    topology=raw.get("topology"),
                    params=raw.get("params", {}),
                )

            elif kind == "boolean":
                op = BooleanOp(
                    boolean_kind=BooleanKind(raw["boolean_kind"]),
                    operands=raw.get("operands"),
                    params=raw.get("params", {}),
                )

            elif kind == "export":
                op = ExportOp(
                    format=raw.get("format", "step"),
                    path=raw.get("path", "output.step"),
                )

            else:
                raise TorqueCLIError(f"Unknown IR op kind '{kind}'")

            ops.append(op)

        except Exception as e:
            raise TorqueCLIError(
                f"Failed to parse IR op at index {idx}: {e}"
            ) from e

    return IRModel(ops)


def main():
    parser = argparse.ArgumentParser(
        prog="torque",
        description="Torque: IR -> CAD compiler",
    )

    parser.add_argument(
        "ir_file",
        type=Path,
        help="Path to Lexica IR JSON file",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output STEP file path (must be consumed by export op)",
    )

    args = parser.parse_args()

    ir_path = args.ir_file.resolve()
    if not ir_path.exists():
        raise TorqueCLIError(f"IR file not found: {ir_path}")

    # ---------------------------------
    # Load + validate IR
    # ---------------------------------
    ir = load_ir(ir_path)
    validate_ir(ir)

    # ---------------------------------
    # Lower IR -> IRL
    # ---------------------------------
    irl = lower_ir_to_irl(ir)

    # ---------------------------------
    # Patch export path (CLI-level policy)
    # ---------------------------------
    if args.out is not None:
        out_path = str(args.out.resolve())
        for op in irl.ops:
            if op.category.value == "export":
                op.params["path"] = out_path

    # ---------------------------------
    # Execute IRL
    # ---------------------------------
    run_irl(irl)


if __name__ == "__main__":
    main()
