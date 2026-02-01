"""
End-to-end evaluation of Orbit + Torque.

This module tests the full Lexica stack:
NL --> Orbit --> IR --> IRL --> Kernel
"""

from pathlib import Path
from typing import Any

from lexica.orbit.pipeline import run_orbit

from lexica.torque.language.ir.validate import validate_ir
from lexica.torque.irl.ir_to_irl import lower_ir_to_irl
from lexica.torque.irl.validation import validate_irl
from lexica.torque.kernel.executor import IRLExecutor


class OrbitEvaluationError(Exception):
    """Raised when Orbit evaluation fails."""
    pass


def run_kernel_from_prompt(
    prompt: str,
    output_path: str | Path,
    **executor_kwargs: Any,
) -> None:
    """
    Run the full pipeline from natural language to CAD output.

    Args:
        prompt: Natural language modeling instruction
        output_path: Path to output STEP file
        executor_kwargs: Optional kernel execution parameters
    """
    try:
        # 1. Orbit: NL -> IR
        ir = run_orbit(prompt)

        # 2. IR validation (explicit, defensive)
        validate_ir(ir)

        # 3. Lower IR -> IRL
        irl = lower_ir_to_irl(ir)

        # 4. IRL validation
        validate_irl(irl)

        # 5. Kernel execution
        executor = IRLExecutor(**executor_kwargs)
        executor.execute(irl, output_path=str(output_path))

    except Exception as e:
        raise OrbitEvaluationError(
            f"Orbit evaluation failed: {e}"
        ) from e


if __name__ == "__main__":
    # Simple manual test
    out = Path("orbit_eval_output.step")
    run_kernel_from_prompt(
        "Create a box of 10 by 20 by 5 millimeters",
        out,
    )
    print(f"Output written to {out.resolve()}")
