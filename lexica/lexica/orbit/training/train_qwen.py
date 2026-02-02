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


# ============================================================
# HARD CPU LIMITS 
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
# PATHS / CONSTANTS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ORBIT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ORBIT_ROOT / "dataset"
OUT_DIR = ORBIT_ROOT / "artifacts" / "orbit-qwen2.5-1.5b-lora"

TRAIN_FILE = DATA_DIR / "train.jsonl"

MAX_SEQ_LEN = 512
SEED = 1337


# ============================================================
# DATASET
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


def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    out["labels"] = out["input_ids"]
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    global tokenizer

    print("[cpu-train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[cpu-train] Loading dataset...")
    train_ds = build_dataset(TRAIN_FILE, tokenizer)

    print("[cpu-train] Tokenizing (CPU-parallel)...")
    train_ds = train_ds.map(
        tokenize,
        remove_columns=["text"],
        num_proc=CPU_CORES,
        desc="Tokenizing",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("[cpu-train] Loading model (fp32 CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # fp32 is safer on generic CPUs
        low_cpu_mem_usage=True,
        device_map=None,
    )

    print("[cpu-train] Applying LoRA...")
    lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        seed=SEED,

        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,   # effective batch = 32

        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,

        logging_steps=25,
        save_steps=500,
        save_total_limit=2,

        report_to="none",
        evaluation_strategy="no",

        dataloader_num_workers=0,   # CRITICAL: faster on small CPUs
        dataloader_pin_memory=False,

        optim="adamw_torch",
        use_cpu=True,

        fp16=False,
        bf16=False,                 # enable if CPU supports it

        gradient_checkpointing=True,
        torch_compile=False,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("[cpu-train] Starting training...")
    trainer.train()

    print("[cpu-train] Saving artifacts...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print("[cpu-train] Done.")


if __name__ == "__main__":
    main()
