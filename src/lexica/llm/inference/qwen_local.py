import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
ADAPTER_PATH = BASE_DIR / "lexica" / "llm" / "artifacts" / "lexica-ir-v1-lora"
SYSTEM_PROMPT_PATH = BASE_DIR / "lexica" / "llm" / "prompts" / "system.txt"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def cut_after_first_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text.strip()

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
                return text[start:i+1].strip()

    return text[start:].strip()


# Load model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
model.eval()

system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def generate(user_prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_text = decoded[len(prompt):].strip()
    return cut_after_first_json(assistant_text)
