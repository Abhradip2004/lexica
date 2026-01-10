from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = Path(__file__).parents[1] / "artifacts" / "lexica-ir-v1-lora"


class Orbit:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map=None,  # CPU
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
        )

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return text
