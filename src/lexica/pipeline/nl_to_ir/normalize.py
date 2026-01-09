"""
IR Normalization Pass
====================

Purpose:
- Convert LLM-produced IR into kernel-valid IR
- Apply semantic defaults
- Complete underspecified topology
- Reject or clean up hallucinated parameters

This module:
- DOES NOT talk to the kernel
- DOES NOT execute geometry
- DOES NOT loosen validation rules
"""

from copy import deepcopy

from lexica.pipeline.nl_to_ir.schema import (
    IRModel,
    FeatureKind,
)

from lexica.pipeline.nl_to_ir.validate import IRValidationError

from lexica.irl.contract import TopoTarget


def normalize_ir(ir: IRModel) -> IRModel:
    """
    Normalize an IRModel produced by the LLM.

    Normalization rules (MVP):
    1. 'hole' defaults to through-hole
    2. Blind holes must specify depth
    3. Face topology min/max defaults to Z axis
    4. Hallucinated center offsets are removed
    """

    ir = deepcopy(ir)

    for idx, op in enumerate(ir.ops):

        # -------------------------------------------------
        # Only feature ops need normalization
        # -------------------------------------------------
        if not hasattr(op, "feature_kind"):
            continue

        # =================================================
        # HOLE normalization
        # =================================================
        if op.feature_kind == FeatureKind.HOLE:
            params = op.params

            has_depth = "depth" in params
            has_through = "through_all" in params

            # Rule 1: "hole" means through-hole by default
            if not has_depth and not has_through:
                params["through_all"] = True

            # Rule 2: Blind hole must specify depth
            if params.get("through_all") is False and not has_depth:
                raise IRValidationError(
                    f"Hole op {idx} requires depth or through_all=true"
                )

            # Rule 3: Remove hallucinated center offsets
            # MVP: only (0,0) is allowed implicitly
            if "center" in params:
                cx, cy = params["center"]
                if cx != 0 or cy != 0:
                    del params["center"]

        # =================================================
        # TOPOLOGY normalization
        # =================================================
        topo = getattr(op, "topology", None)
        if topo is not None and topo.target == TopoTarget.FACE:

            # Rule 4: min/max face defaults to vertical axis
            if topo.rule in ("min", "max") and topo.value is None:
                topo.value = "Z"

    return ir
