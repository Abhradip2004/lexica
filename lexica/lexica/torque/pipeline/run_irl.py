"""
IRL Execution Pipeline

This module executes a validated IRL program using the Torque kernel.

This is a PURE execution step:
- No lowering
- No inference
- No defaults
- No recovery
"""

from lexica.torque.irl.validation import validate_irl
from lexica.torque.kernel.executor import IRLExecutor


class IRLRunError(Exception):
    pass


def run_irl(irl_model):
    """
    Execute a fully-formed IRL program.

    Args:
        irl_model (IRLModel): Valid IRL program

    Raises:
        IRLRunError: if validation or execution fails
    """

    try:
        # ----------------------------------
        # Validate IRL
        # ----------------------------------
        validate_irl(irl_model)

        # ----------------------------------
        # Execute kernel
        # ----------------------------------
        executor = IRLExecutor()
        executor.execute(irl_model)

    except Exception as e:
        raise IRLRunError(f"IRL execution failed: {e}") from e
