import cadquery as cq
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# -----------------------------
# IR Definitions -v0
# -----------------------------

@dataclass(frozen=True)
class IROp:
    op: str
    params: Dict[str, Any]


# -----------------------------
# Kernel Result
# -----------------------------

@dataclass
class KernelResult:
    solid: Optional[object]
    errors: List[str]
    warnings: List[str]


# -----------------------------
# CadQuery Kernel
# -----------------------------

class CadQueryKernel:
    """
    Lexica CadQuery kernel
    --------------------
    - Executes IR operations sequentially
    - Deterministic
    - No inference, no intelligence
    """

    def __init__(self):
        self._dispatch = {
            "box": self._op_box,
        }

    # ---------
    # Public API
    # ---------

    def execute(self, ir_ops: List[IROp]) -> KernelResult:
        wp = cq.Workplane("XY")
        errors: List[str] = []
        warnings: List[str] = []

        for idx, op in enumerate(ir_ops):
            try:
                wp = self._execute_op(wp, op)
            except Exception as e:
                errors.append(
                    f"IR op {idx} '{op.op}' failed: {str(e)}"
                )
                return KernelResult(
                    solid=None,
                    errors=errors,
                    warnings=warnings
                )

        # Final solid validation
        try:
            solid = wp.val()
        except Exception as e:
            return KernelResult(
                solid=None,
                errors=[f"Final solid invalid: {str(e)}"],
                warnings=warnings
            )

        return KernelResult(
            solid=solid,
            errors=errors,
            warnings=warnings
        )

    # -----------------
    # Internal Dispatch
    # -----------------

    def _execute_op(self, wp: cq.Workplane, op: IROp) -> cq.Workplane:
        if op.op not in self._dispatch:
            raise ValueError(f"Unknown IR operation: {op.op}")

        return self._dispatch[op.op](wp, op.params)

    # -----------------
    # IR Operations
    # -----------------

    def _op_box(self, wp: cq.Workplane, params: Dict[str, Any]) -> cq.Workplane:
        """
        IR op: box
        params:
            - x: float
            - y: float
            - z: float
        """
        try:
            x = float(params["x"])
            y = float(params["y"])
            z = float(params["z"])
        except KeyError as e:
            raise ValueError(f"Missing box parameter: {e}")
        except ValueError:
            raise ValueError("Box parameters must be numeric")

        return wp.box(x, y, z)
