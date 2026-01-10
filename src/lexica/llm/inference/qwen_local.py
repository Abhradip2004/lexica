from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]
ADAPTER_PATH = BASE_DIR / "lexica/llm/artifacts/lexica-ir-v1-lora"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# -------------------------------------------------
# Load model
# -------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
model.eval()

# -------------------------------------------------
# Inference
# -------------------------------------------------

def generate(prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
