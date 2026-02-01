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

MAX_SEQ_LEN = 1024
SEED = 1337

CPU_CORES = os.cpu_count() or 32


# -------------------------------------------------
# Dataset
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
        padding="max_length",  # IMPORTANT for CPU
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    torch.set_num_threads(CPU_CORES)
    torch.set_num_interop_threads(min(8, CPU_CORES))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(TRAIN_PATH, tokenizer)
    eval_ds = build_dataset(EVAL_PATH, tokenizer)

    train_ds = train_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=min(8, CPU_CORES),
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_fn(tokenizer, x),
        remove_columns=["text"],
        num_proc=min(8, CPU_CORES),
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

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

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        seed=SEED,

        num_train_epochs=5,
        per_device_train_batch_size=4,      # CPU-friendly
        gradient_accumulation_steps=2,

        learning_rate=2e-4,
        warmup_steps=200,

        logging_steps=10,
        save_steps=500,
        save_total_limit=2,

        dataloader_num_workers=min(16, CPU_CORES),
        dataloader_prefetch_factor=4,

        report_to="none",
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=200,

        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
