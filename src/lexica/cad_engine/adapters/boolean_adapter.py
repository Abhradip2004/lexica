from __future__ import annotations

import cadquery as cq

from lexica.irl.contract import BooleanOp, BooleanKind


class BooleanAdapterError(Exception):
    pass


def execute_boolean(op: BooleanOp, shapes: list):
    if op.kind == BooleanKind.UNION:
        return _union(shapes)

    if op.kind == BooleanKind.DIFFERENCE:
        return _difference(shapes)

    if op.kind == BooleanKind.INTERSECTION:
        return _intersection(shapes)

    raise BooleanAdapterError(f"Unknown boolean kind: {op.kind}")


def _union(shapes):
    wp = cq.Workplane(obj=shapes[0])
    for shape in shapes[1:]:
        wp = wp.union(shape)
    return wp.val()


def _difference(shapes):
    wp = cq.Workplane(obj=shapes[0])
    for shape in shapes[1:]:
        wp = wp.cut(shape)
    return wp.val()


def _intersection(shapes):
    wp = cq.Workplane(obj=shapes[0])
    for shape in shapes[1:]:
        wp = wp.intersect(shape)
    return wp.val()
