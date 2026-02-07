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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# GPU CONFIG (RTX 3090)
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

assert torch.cuda.is_available(), "CUDA is required for this script"
DEVICE = "cuda"


# ============================================================
# PATHS
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ORBIT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ORBIT_ROOT / "dataset"
ARTIFACTS_DIR = ORBIT_ROOT / "artifacts" / "orbit-qwen2.5-1.5b-qlora"

TRAIN_FILE = DATASET_DIR / "train.jsonl"

MAX_SEQ_LEN = 2048
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


def tokenize_fn(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# ============================================================
# MAIN
# ============================================================

def main():
    assert TRAIN_FILE.exists(), f"Missing dataset file: {TRAIN_FILE}"

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Dataset ----------------
    train_ds = build_dataset(TRAIN_FILE, tokenizer)
    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["text"],
        num_proc=4,
    )

    # ---------------- QLoRA Quantization ----------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------- Model ----------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()

    # ---------------- LoRA ----------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- Training args ----------------
    args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR),
        seed=SEED,

        num_train_epochs=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,

        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,

        bf16=True,
        fp16=False,

        optim="paged_adamw_8bit",
        gradient_checkpointing=True,

        report_to="none",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=None,  # HF handles causal LM padding correctly here
    )

    # ---------------- Train ----------------
    trainer.train()

    # ---------------- Save ----------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ARTIFACTS_DIR))
    tokenizer.save_pretrained(str(ARTIFACTS_DIR))


if __name__ == "__main__":
    main()
