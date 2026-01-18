import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# -------------------------------------------------
# Config
# -------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parents[3]

TRAIN_PATH = BASE_DIR / "lexica" / "llm" / "dataset" / "train.jsonl"
EVAL_PATH  = BASE_DIR / "lexica" / "llm" / "dataset" / "eval.jsonl"

# This must match inference loaders:
# - load_model.py expects lexica/llm/artifacts/lexica-ir-v1-lora
# - qwen_local.py expects lexica/llm/artifacts/lexica-ir-v1-lora
OUTPUT_DIR = BASE_DIR / "lexica" / "llm" / "artifacts" / "lexica-ir-v1-lora"

MAX_SEQ_LEN = 1024

torch.set_num_threads(4)


# -------------------------------------------------
# Dataset loading
# -------------------------------------------------

def load_jsonl_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_conversation(example: dict) -> str:
    """
    Convert OpenAI-style messages into Qwen chat format.
    """
    out = []
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        out.append(f"<|{role}|>\n{content}")
    return "\n".join(out)


def build_dataset(path: Path) -> Dataset:
    raw = load_jsonl_dataset(path)
    texts = [format_conversation(ex) for ex in raw]
    return Dataset.from_dict({"text": texts})


# -------------------------------------------------
# Tokenization
# -------------------------------------------------

def tokenize_fn(tokenizer, example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    # Optional: force offline to avoid HF read timeouts
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print("[train] loading datasets...")
    train_ds = build_dataset(TRAIN_PATH)
    eval_ds = build_dataset(EVAL_PATH)

    print("[train] train size:", len(train_ds))
    print("[train] eval size :", len(eval_ds))
    print("[train] sample:\n", train_ds[0]["text"][:400], "...")

    print("[train] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,  # CPU
        trust_remote_code=True,
        local_files_only=True,
    )

    print("[train] applying LoRA...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[train] tokenizing...")
    train_ds = train_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,

        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-4,

        logging_steps=10,
        save_steps=100,
        save_total_limit=1,

        fp16=False,
        bf16=False,

        report_to="none",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("[train] starting training...")
    trainer.train()

    print("[train] saving adapter...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))

    print("[train] done.")
    print("[train] adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
