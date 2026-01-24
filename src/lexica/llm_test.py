import json

from lexica.llm.inference.load_model import Orbit

from lexica.pipeline.nl_to_ir.schema import IRModel
from lexica.pipeline.nl_to_ir.validate import validate_ir

from lexica.irl.ir_to_irl import lower_ir_to_irl
from lexica.irl.validation import validate_irl
from lexica.cad_engine.executor import IRLExecutor

def extract_first_json_object(text: str) -> str:
    """
    Extract first valid {...} JSON object substring using brace balancing.
    Works even if the model prints extra junk before/after.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in output")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                # sanity check: must be parseable JSON
                json.loads(candidate)
                return candidate

    raise ValueError("No complete JSON object found (unbalanced braces)")


def run_kernel_from_ir_dict(ir_dict):
    # dict -> dataclass
    ir_model = IRModel(**{
        "ops": ir_dict["ops"]
    })

    validate_ir(ir_model)
    irl_model = lower_ir_to_irl(ir_model)
    validate_irl(irl_model)

    ex = IRLExecutor()
    ex.execute(irl_model)


def main():
    orbit = Orbit()

    prompt = "Create a rectangular mounting plate 120mm by 80mm by 6mm. Fillet edges 2mm. Add four corner through holes of 6mm diameter. Add a through hole in the center of 30mm Export step."
    print("PROMPT:\n", prompt)

    out = orbit.generate(prompt)
    print("\nORBIT OUTPUT:\n", out)

    # parse json
    ir_dict = json.loads(extract_first_json_object(out))

    # run kernel
    run_kernel_from_ir_dict(ir_dict)

    print("\nSUCCESS: kernel executed. Check lexica_output.step")


if __name__ == "__main__":
    main()
