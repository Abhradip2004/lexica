from __future__ import annotations

import cadquery as cq

from lexica.irl.contract import ExportOp


class ExportAdapterError(Exception):
    pass


def execute_export(op: ExportOp, shape):
    fmt = op.params.get("format")
    path = op.params.get("path")

    if not fmt or not path:
        raise ExportAdapterError("Export requires format and path")

    if fmt.lower() == "step":
        cq.exporters.export(shape, path)
        return

    raise ExportAdapterError(f"Unsupported export format: {fmt}")
