import os
import json
from tqdm import tqdm

from lexica.pipeline.nl_to_ir.validate import validate_ir
from lexica.irl.ir_to_irl import lower_ir_to_irl
from lexica.irl.validation import validate_irl
from lexica.cad_engine.executor import IRLExecutor

from lexica.llm.inference.load_model import Orbit


# ----------------------------
# Path-safe config
# ----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
LEXICA_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  

EVAL_FILE = os.path.join(LEXICA_SRC_DIR, "llm", "dataset", "eval.jsonl")


def run_kernel(ir_dict):
    try:
        # reuse your dict_to_ir_model logic here
        from lexica.llm.dataset.evaluate_dataset import dict_to_ir_model
        ir = dict_to_ir_model(ir_dict)

        validate_ir(ir)
        irl = lower_ir_to_irl(ir)
        validate_irl(irl)

        IRLExecutor().execute(irl)
        return True
    except Exception:
        return False


def main():
    orbit = Orbit()

    total = 0
    kernel_valid = 0

    with open(EVAL_FILE) as f:
        for line in tqdm(f):
            obj = json.loads(line)

            user_prompt = obj["messages"][1]["content"]
            gt_ir = json.loads(obj["messages"][2]["content"])

            pred_text = orbit.generate(user_prompt)

            try:
                pred_ir = json.loads(pred_text)
            except Exception:
                total += 1
                continue

            if run_kernel(pred_ir):
                kernel_valid += 1

            total += 1

    print("Total samples:", total)
    print("Kernel-valid:", kernel_valid)
    print("Kernel-valid accuracy:", kernel_valid / total)


if __name__ == "__main__":
    main()
