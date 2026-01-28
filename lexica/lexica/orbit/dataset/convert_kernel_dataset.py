import os
import json
import random
from typing import Dict, Any, List

# ----------------------------
# Path-safe config (works from any cwd)
# ----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))         # .../src/lexica/llm/dataset
LEXICA_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # .../src/lexica

INPUT_FILE = os.path.join(LEXICA_SRC_DIR, "final_dataset.jsonl")
SYSTEM_PROMPT_FILE = os.path.join(LEXICA_SRC_DIR, "llm", "prompts", "system.txt")

OUT_DIR = os.path.join(LEXICA_SRC_DIR, "llm", "dataset")
TRAIN_OUT = os.path.join(OUT_DIR, "train.jsonl")
EVAL_OUT = os.path.join(OUT_DIR, "eval.jsonl")

EVAL_RATIO = 0.12
SEED = 42



# ----------------------------
# Helpers
# ----------------------------

def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def ir_to_pretty_json(ir_dict: Dict[str, Any]) -> str:
    # Orbit should output IR JSON as assistant text
    # Using compact json is also fine, but pretty json helps training readability.
    return json.dumps(ir_dict, indent=2, sort_keys=True)


def make_messages(system_prompt: str, user_prompt: str, ir_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ir_to_pretty_json(ir_dict)},
        ]
    }


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)

    samples: List[Dict[str, Any]] = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            prompt = obj.get("prompt")
            ir = obj.get("ir")

            if not prompt or not ir:
                continue

            samples.append(obj)

    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")

    # Shuffle then split
    random.shuffle(samples)
    n_eval = int(len(samples) * EVAL_RATIO)

    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    print(f"Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    # Write outputs
    with open(TRAIN_OUT, "w", encoding="utf-8") as ftrain:
        for obj in train_samples:
            rec = make_messages(system_prompt, obj["prompt"], obj["ir"])
            ftrain.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(EVAL_OUT, "w", encoding="utf-8") as feval:
        for obj in eval_samples:
            rec = make_messages(system_prompt, obj["prompt"], obj["ir"])
            feval.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Conversion complete")
    print(f"Saved train dataset --> {TRAIN_OUT}")
    print(f"Saved eval dataset  --> {EVAL_OUT}")


if __name__ == "__main__":
    main()
