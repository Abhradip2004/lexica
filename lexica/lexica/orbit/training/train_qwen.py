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
)
from peft import LoraConfig, get_peft_model


# ============================================================
# CPU CONFIG (4 vCPU VPS / laptop safe)
# ============================================================

CPU_CORES = 4

torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(1)

os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print(f"[cpu-train] Using {CPU_CORES} CPU threads")


# ============================================================
# PATHS
# orbit/
# ├── dataset/
# │   └── train.jsonl
# ├── train/
# │   └── train_qwen.py  ← this file
# └── artifacts/
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ORBIT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ORBIT_ROOT / "dataset"
ARTIFACTS_DIR = ORBIT_ROOT / "artifacts" / "orbit-qwen2.5-1.5b-lora"

TRAIN_FILE = DATASET_DIR / "train.jsonl"

MAX_SEQ_LEN = 512
SEED = 1337


# ============================================================
# DATASET UTILITIES
# ============================================================

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_dataset(path: Path, tokenizer):
    texts = []
    for ex in load_jsonl(path):
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def tokenize_fn(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    tokens["labels"] = tokens["input_ids"]
    return tokens


# ============================================================
# MAIN
# ============================================================

def main():
    assert TRAIN_FILE.exists(), f"Missing dataset file: {TRAIN_FILE}"

    # ---------------- Tokenizer ----------------
    print("[cpu-train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Dataset ----------------
    print("[cpu-train] Loading dataset...")
    train_ds = build_dataset(TRAIN_FILE, tokenizer)

    print("[cpu-train] Tokenizing...")
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["text"],
        num_proc=CPU_CORES,
    )

    # ---------------- Custom collator (CORRECT & FINAL) ----------------
    def causal_lm_collator(features):
        # Extract fields
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Find max length in batch
        max_len = max(len(x) for x in input_ids)

        # Pad input_ids, attention_mask, labels
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for ids, mask, lbl in zip(input_ids, attention_masks, labels):
            pad_len = max_len - len(ids)

            padded_input_ids.append(
                ids + [tokenizer.pad_token_id] * pad_len
            )
            padded_attention_masks.append(
                mask + [0] * pad_len
            )
            padded_labels.append(
                lbl + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


    # ---------------- Model ----------------
    print("[cpu-train] Loading model (BF16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # ---------------- LoRA ----------------
    print("[cpu-train] Applying LoRA...")
    lora_config = LoraConfig(
        r=2,                 # safe for 16 GB RAM
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- Training arguments ----------------
    args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR),
        seed=SEED,

        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,

        logging_steps=25,
        save_steps=500,
        save_total_limit=2,

        report_to="none",
        eval_strategy="no",

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        optim="adamw_torch",
        use_cpu=True,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        torch_compile=False,
        remove_unused_columns=True,
    )

    # ---------------- Trainer ----------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=causal_lm_collator,
    )

    # ---------------- Train ----------------
    print("[cpu-train] Starting training...")
    trainer.train()

    # ---------------- Save ----------------
    print("[cpu-train] Saving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ARTIFACTS_DIR))
    tokenizer.save_pretrained(str(ARTIFACTS_DIR))

    print("[cpu-train] Done.")


if __name__ == "__main__":
    main()
