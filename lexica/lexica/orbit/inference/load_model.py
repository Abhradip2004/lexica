from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR = Path(__file__).resolve().parents[3]
ADAPTER_PATH = BASE_DIR / "lexica" / "llm" / "artifacts" / "lexica-ir-v1-lora"
SYSTEM_PROMPT_PATH = BASE_DIR / "lexica" / "llm" / "prompts" / "system.txt"


def cut_after_first_json(text: str) -> str:
    """
    Extract first complete JSON object using brace balancing.
    Returns original text if '{' not found.
    """
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

    # If the model never closed braces, return partial JSON tail
    return text[start:].strip()


class Orbit:
    def __init__(self):
        # Load system prompt
        if not SYSTEM_PROMPT_PATH.exists():
            raise FileNotFoundError(f"Missing system prompt: {SYSTEM_PROMPT_PATH}")

        self.system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        # Base model (CPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

        # LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        self.model.eval()

    def generate(self, user_prompt: str, max_new_tokens: int = 512) -> str:
        """
        Natural user prompt -> model -> returns ONLY the first JSON object (if any).
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # compiler-like deterministic decoding
                repetition_penalty=1.05,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # take only assistant part
        assistant_text = decoded[len(prompt):].strip()

        # force only JSON
        return cut_after_first_json(assistant_text)
