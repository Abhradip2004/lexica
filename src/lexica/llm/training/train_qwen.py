import json
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

DATASET_PATH = (
    BASE_DIR
    / "lexica"
    / "llm"
    / "dataset"
    / "box_plus_features.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "lexica"
    / "llm"
    / "checkpoints"
    / "qwen-box"
)

MAX_SEQ_LEN = 1024

# CPU-safe
torch.set_num_threads(4)


# -------------------------------------------------
# Dataset loading
# -------------------------------------------------

def load_jsonl_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r") as f:
        for line in f:
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
    print("[train] loading dataset...")
    train_ds = build_dataset(DATASET_PATH)

    print("[train] dataset size:", len(train_ds))
    print("[train] sample:\n", train_ds[0]["text"][:400], "...")

    print("[train] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
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

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,

        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        learning_rate=2e-4,

        logging_steps=1,
        save_steps=50,
        save_total_limit=1,

        fp16=False,
        bf16=False,

        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    print("[train] starting training...")
    trainer.train()

    print("[train] saving adapter...")
    trainer.save_model()

    print("[train] done.")


if __name__ == "__main__":
    main()
