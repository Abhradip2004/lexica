import pytest

from lexica.pipeline.nl_to_ir.translator import Translator
from lexica.pipeline.nl_to_ir.schema import IR
from lexica.pipeline.nl_to_ir.validate import IRValidationError


class MockLLM:
    def generate_structured(self, prompt: str, schema: dict):
        return {
            "entity": "solid",
            "primitive": "box",
            "parameters": {
                "length": 40,
                "width": 30,
                "height": 20,
            },
            "units": "mm",
        }


def test_translate_box_success():
    translator = Translator(MockLLM())

    ir = translator.translate(
        "Create a box of 40 by 30 by 20 millimeters"
    )

    assert isinstance(ir, IR)
    assert ir.primitive == "box"
    assert ir.parameters["length"] == 40
    assert ir.units == "mm"


def test_invalid_parameter_rejected():
    class BadLLM:
        def generate_structured(self, prompt, schema):
            return {
                "entity": "solid",
                "primitive": "box",
                "parameters": {
                    "length": -10,
                    "width": 30,
                    "height": 20,
                },
                "units": "mm",
            }

    translator = Translator(BadLLM())

    with pytest.raises(IRValidationError):
        translator.translate("Bad box")
