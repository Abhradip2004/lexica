from __future__ import annotations

import cadquery as cq

from lexica.torque.irl.contract import ExportOp


class ExportAdapterError(Exception):
    pass


from pathlib import Path

def execute_export(op: ExportOp, shape):
    """
    Execute an export operation.

    MVP behavior:
    - Export format is required
    - Output file is always written to the PROJECT ROOT
    - Custom paths are intentionally NOT supported
    """

    params = op.params

    # -------------------------------------------------
    # Validate format
    # -------------------------------------------------
    fmt = params.get("format")
    if not fmt:
        raise ExportAdapterError(
            "Export operation requires 'format' (e.g. 'step')"
        )

    fmt = fmt.lower()

    # -------------------------------------------------
    # Resolve PROJECT ROOT explicitly
    # -------------------------------------------------
    # This file lives at:
    #   lexica/cad_engine/adapters/export_adapter.py
    # So project root = 4 levels up
    output = params.get("path")
    if not output:
        raise ExportAdapterError(
            "Export operation requires 'path' parameter"
        )

    output_path = Path(output).resolve()

    # -------------------------------------------------
    # Dispatch export
    # -------------------------------------------------
    if fmt == "step":
        cq.exporters.export(shape, str(output_path))
        return

    raise ExportAdapterError(
        f"Unsupported export format '{fmt}'. "
        "Currently supported: step"
    )

