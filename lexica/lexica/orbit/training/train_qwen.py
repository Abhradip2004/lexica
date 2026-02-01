"""
Orbit fine-tuning script (CPU BF16 path, optimized for AMD EPYC 9354P).

Target:
Natural Language --> Lexica IR (JSON)

Hardware target:
- AMD EPYC 9354P (32 cores/64 threads)
- CPU-only
- <= 16 GB RAM
- Transformers v5
"""

import os
import json
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

# Optional: For monitoring CPU/RAM usage (install if needed: pip install psutil)
import psutil

# -------------------------------------------------------------------
# CPU threading (optimized for EPYC 9354P)
# -------------------------------------------------------------------

num_cores = os.cpu_count()  # Should be 64 on your setup
torch.set_num_threads(num_cores // 2)  # 32 for intra-op (balance compute vs. overhead)
torch.set_num_interop_threads(num_cores // 4)  # 16 for inter-op

print(f"[orbit-train] Detected {num_cores} logical cores. Set intra-op threads: {torch.get_num_threads()}, inter-op: {torch.get_num_interop_threads()}")

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORBIT_DIR = PROJECT_ROOT / "lexica" / "orbit"
DATASET_DIR = ORBIT_DIR / "dataset"

TRAIN_FILE = DATASET_DIR / "train.jsonl"
EVAL_FILE = DATASET_DIR / "eval.jsonl"

ARTIFACTS_DIR = ORBIT_DIR / "artifacts" / "orbit-ir-v0-lora"

MAX_SEQ_LEN = 512
SEED = 1337

# -------------------------------------------------------------------
# Dataset utilities
# -------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def build_dataset(path: Path, tokenizer: AutoTokenizer) -> Dataset:
    raw = load_jsonl(path)

    texts = []
    for ex in raw:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})

def tokenize_fn(tokenizer, example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,   # dynamic padding via collator
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print("[orbit-train] BF16 CPU path enabled (optimized for EPYC 9354P)")

    # ---------------- Tokenizer ----------------

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Dataset ----------------

    train_ds = build_dataset(TRAIN_FILE, tokenizer)
    eval_ds = build_dataset(EVAL_FILE, tokenizer)

    train_ds = train_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=16,  # Increased for parallelism
    )

    eval_ds = eval_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=16,  # Increased for parallelism
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ---------------- Model (BF16 end-to-end) ----------------

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,   # IMPORTANT
        trust_remote_code=True,
    )

    # ---------------- LoRA ----------------

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

    # ---------------- Training arguments ----------------

    training_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR),
        seed=SEED,

        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,

        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,

        logging_steps=50,

        save_steps=500,
        save_total_limit=2,

        report_to="none",
        remove_unused_columns=True,

        eval_strategy="no",

        dataloader_num_workers=16,  # Increased for better I/O parallelism
        dataloader_pin_memory=True,  # Enable for faster data transfer (CPU-safe)
        
        use_cpu=True,

        optim="adamw_torch",

        fp16=False,
        bf16=True,   # IMPORTANT
    )

    # ---------------- Trainer ----------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print("[orbit-train] starting training (BF16)...")
    trainer.train()

    # Optional: Log final hardware usage
    print(f"[orbit-train] CPU usage: {psutil.cpu_percent()}% | RAM usage: {psutil.virtual_memory().percent}%")

    # ---------------- Save ----------------

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ARTIFACTS_DIR))
    tokenizer.save_pretrained(ARTIFACTS_DIR)

    print("[orbit-train] done.")
    print("[orbit-train] artifacts saved to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    main()