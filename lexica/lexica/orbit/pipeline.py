"""
Orbit pipeline.

Natural language --> Lexica IR
"""

from lexica.orbit.translator import translate_to_ir
from lexica.torque.language.ir.schema import IRModel


class OrbitPipelineError(Exception):
    """Raised when Orbit fails to produce valid IR."""
    pass


def run_orbit(prompt: str) -> IRModel:
    """
    Run the full Orbit pipeline.

    Input:
        prompt: plain English modeling instruction

    Output:
        IRModel (validated Lexica IR)

    Raises:
        OrbitPipelineError if IR generation or validation fails
    """
    try:
        ir = translate_to_ir(prompt)
    except Exception as e:
        raise OrbitPipelineError(
            f"Orbit failed to translate prompt to IR: {e}"
        ) from e

    if not isinstance(ir, IRModel):
        raise OrbitPipelineError(
            "Orbit pipeline did not return an IRModel"
        )

    return ir
