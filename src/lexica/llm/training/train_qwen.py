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

# src/lexica/llm/training/train_qwen.py -> parents[3] = src
BASE_DIR = Path(__file__).resolve().parents[3]

TRAIN_PATH = BASE_DIR / "lexica" / "llm" / "dataset" / "train.jsonl"
EVAL_PATH = BASE_DIR / "lexica" / "llm" / "dataset" / "eval.jsonl"

OUTPUT_DIR = BASE_DIR / "lexica" / "llm" / "artifacts" / "lexica-ir-v1-lora"

MAX_SEQ_LEN = 1024
SEED = 1337

torch.set_num_threads(4)


# -------------------------------------------------
# Dataset loading
# -------------------------------------------------

def load_jsonl_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_dataset(path: Path, tokenizer: AutoTokenizer) -> Dataset:
    """
    Converts OpenAI-style `messages` into the model's *official*
    chat template using `tokenizer.apply_chat_template`.
    """
    raw = load_jsonl_dataset(path)

    texts = []
    for ex in raw:
        msgs = ex["messages"]

        # Important: We must NOT add a generation prompt during training.
        # We want the assistant message included in the training text.
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

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
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print("[train] train file:", TRAIN_PATH)
    print("[train] eval file :", EVAL_PATH)

    print("[train] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=False,
    )

    # Needed for collator padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] loading datasets...")
    train_ds = build_dataset(TRAIN_PATH, tokenizer)
    eval_ds = build_dataset(EVAL_PATH, tokenizer)

    print("[train] train size:", len(train_ds))
    print("[train] eval size :", len(eval_ds))
    print("[train] sample preview:\n", train_ds[0]["text"][:500], "...")

    print("[train] tokenizing dataset...")
    train_ds = train_ds.map(lambda x: tokenize_fn(tokenizer, x), remove_columns=["text"])
    eval_ds = eval_ds.map(lambda x: tokenize_fn(tokenizer, x), remove_columns=["text"])

    # pads labels with -100
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_cuda = torch.cuda.is_available()
    print("[train] cuda available:", use_cuda)
    if use_cuda:
        print("[train] gpu:", torch.cuda.get_device_name(0))

    dtype = torch.float16 if use_cuda else torch.float32

    print("[train] loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        local_files_only=False,
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

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        seed=SEED,

        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,

        logging_steps=10,

        save_steps=500,
        save_total_limit=2,

        report_to="none",
        remove_unused_columns=False,

        eval_strategy="steps",
        eval_steps=200,

        fp16=bool(use_cuda),
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
