"""
Orbit fine-tuning script (CPU-optimized).

Target:
Natural Language --> Lexica IR (JSON)

This version is optimized for:
- CPU-only training
- AMD EPYC (Zen 4)
- Memory stability (no OOM)
- Reasonable speed without GPU

Most optimization-related lines are commented so they can be tweaked later.
"""

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
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# -------------------------------------------------
# Config
# -------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = BASE_DIR / "lexica" / "orbit" / "dataset"
TRAIN_PATH = DATASET_DIR / "train.jsonl"
EVAL_PATH = DATASET_DIR / "eval.jsonl"

OUTPUT_DIR = (
    BASE_DIR
    / "lexica"
    / "orbit"
    / "artifacts"
    / "orbit-ir-v0-lora"
)

SEED = 1337

# Maximum sequence length.
# Lower values greatly reduce attention cost on CPU.
# If your data allows, try 384 or 256 later.
MAX_SEQ_LEN = 576


# -------------------------------------------------
# CPU / Threading configuration
# -------------------------------------------------

# Do NOT blindly use all available cores.
# 12–16 threads is a safe range for EPYC CPU LLM training.
torch.set_num_threads(16)
torch.set_num_interop_threads(4)


# -------------------------------------------------
# Dataset utilities
# -------------------------------------------------

def load_jsonl_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_dataset(path: Path, tokenizer: AutoTokenizer) -> Dataset:
    raw = load_jsonl_dataset(path)

    texts = [
        tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in raw
    ]

    return Dataset.from_dict({"text": texts})


def tokenize_fn(tokenizer, example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,

        # Dynamic padding is critical on CPU.
        # Static padding causes large RAM spikes.
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("[orbit-train] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[orbit-train] loading datasets...")
    train_ds = build_dataset(TRAIN_PATH, tokenizer)
    eval_ds = build_dataset(EVAL_PATH, tokenizer)

    print("[orbit-train] tokenizing datasets...")

    # num_proc speeds up tokenization only and is safe.
    train_ds = train_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=4,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=4,
    )

    # -------------------------------------------------
    # Model loading
    # -------------------------------------------------

    print("[orbit-train] loading base model...")

    # Use BF16 on CPU.
    # Zen 4 EPYC supports BF16 and it is significantly faster than FP32.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("[orbit-train] applying LoRA...")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,

        # Keep LoRA target modules minimal for CPU training.
        target_modules=["q_proj", "v_proj"],

        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -------------------------------------------------
    # Training arguments (CPU-safe)
    # -------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        seed=SEED,

        num_train_epochs=5,

        # Batch size greater than 1 increases RAM pressure on CPU.
        # per_device_train_batch_size=1,

        # Gradient accumulation improves efficiency without increasing peak memory.
        gradient_accumulation_steps=8,

        learning_rate=2e-4,
        warmup_steps=200,

        logging_steps=50,
        save_steps=500,
        save_total_limit=2,

        # Each DataLoader worker consumes significant RAM.
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,

        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",
        # eval_steps=200,
        use_cpu=True,

        # Precision settings
        bf16=False,
        fp16=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print("[orbit-train] starting training...")
    trainer.train()

    print("[orbit-train] saving adapter...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))

    print("[orbit-train] done.")
    print("[orbit-train] adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
