from .schema import IR


class IRValidationError(Exception):
    pass


SUPPORTED_PRIMITIVES = {"box", "cylinder", "sphere"}


def validate_ir(ir: IR) -> IR:
    if ir.entity != "solid":
        raise IRValidationError("Only solid entities are supported")

    if ir.primitive not in SUPPORTED_PRIMITIVES:
        raise IRValidationError(f"Unsupported primitive: {ir.primitive}")

    if not ir.parameters:
        raise IRValidationError("Parameters cannot be empty")

    for k, v in ir.parameters.items():
        if v <= 0:
            raise IRValidationError(f"Invalid parameter {k}: {v}")

    return ir
