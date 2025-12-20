from .schema import IR
from .validate import validate_ir


class Translator:
    def __init__(self, llm):
        self.llm = llm

    def translate(self, text: str) -> IR:
        result = self.llm.generate_structured(
            prompt=text,
            schema={
                "entity": "solid",
                "primitive": "box | cylinder | sphere",
                "parameters": "dict[str, float]",
                "units": "string",
            },
        )

        ir = IR(
            entity=result["entity"],
            primitive=result["primitive"],
            parameters=result["parameters"],
            units=result.get("units", "mm"),
            constraints=[],
            metadata={"source": "nl_to_ir"},
        )

        return validate_ir(ir)
