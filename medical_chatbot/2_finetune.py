"""
2_finetune.py — QLoRA fine-tuning with Unsloth (Linux / WSL2).

Prerequisites:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

If Unsloth is not available (Windows native), use 2_finetune_fallback.py instead.
"""

import json
import os
import random
import sys

import numpy as np
import torch

# ── Check Unsloth availability ────────────────────────────────
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    UNSLOTH_OK = True
except ImportError:
    print("[ERROR] Unsloth is not installed.")
    print("[INFO]  On Windows native, run: python 2_finetune_fallback.py")
    print("[INFO]  On WSL2/Linux, install with:")
    print('        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    sys.exit(1)

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    BATCH_SIZE,
    BF16,
    FP16,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LR_SCHEDULER,
    MAX_GRAD_NORM,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    OPTIMIZER,
    OUTPUTS_DIR,
    SAVE_STEPS,
    SEED,
    SYSTEM_PROMPT,
    TARGET_MODULES,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

# ── Reproducibility ───────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Helpers ───────────────────────────────────────────────────

def print_vram(label: str):
    """Print current GPU VRAM usage."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] {label}: {used:.2f} GB used / {total:.2f} GB total")
    else:
        print("[VRAM] No GPU detected.")


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run 1_prepare_data.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_conversation(example: dict, tokenizer) -> dict:
    """
    Apply the Qwen 2.5 chat template to a single example.
    Returns a dict with a 'text' field containing the full formatted string.
    """
    messages = [
        {"role": "system",    "content": example.get("system", SYSTEM_PROMPT)},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n=== Step 2: QLoRA Fine-tuning (Unsloth) ===\n")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # 1. Load model and tokenizer
    print(f"[INFO] Loading base model: {BASE_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,           # Auto-detect (BF16 on Ampere, FP16 otherwise)
        load_in_4bit   = True,
        # token = "hf_...",             # Uncomment if model is gated
    )
    print_vram("After model load")

    # 2. Attach LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r               = LORA_R,
        target_modules  = TARGET_MODULES,
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = LORA_DROPOUT,
        bias            = "none",
        use_gradient_checkpointing = "unsloth",  # 30% less VRAM
        random_state    = SEED,
        use_rslora      = False,
        loftq_config    = None,
    )
    print_vram("After LoRA attach")

    # 3. Load and format datasets
    print("[INFO] Loading training data...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)

    train_formatted = [format_conversation(ex, tokenizer) for ex in train_raw]
    val_formatted   = [format_conversation(ex, tokenizer) for ex in val_raw]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset   = Dataset.from_list(val_formatted)
    print(f"[INFO] Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")

    # 4. Configure trainer
    use_bf16 = BF16 and is_bfloat16_supported()
    use_fp16 = FP16 and not use_bf16

    training_args = SFTConfig(
        output_dir                  = ADAPTER_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        optim                       = OPTIMIZER,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        logging_steps               = LOGGING_STEPS,
        save_strategy               = "steps" if SAVE_STEPS > 0 else "epoch",
        save_steps                  = SAVE_STEPS if SAVE_STEPS > 0 else None,
        eval_strategy               = "epoch",
        seed                        = SEED,
        report_to                   = "none",   # Disable wandb / tensorboard
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LENGTH,
        packing                     = False,    # Disable sequence packing for medical Q&A
    )

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        args             = training_args,
    )

    # 5. Train only on assistant responses (not on system/user tokens)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # 6. Train
    print("\n[INFO] Starting training...\n")
    print_vram("Before training")
    trainer.train()
    print_vram("After training")

    # 7. Save LoRA adapter
    print(f"\n[INFO] Saving adapter to: {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print("\n[DONE] Fine-tuning complete.\n")
    print(f"  Adapter saved to : {ADAPTER_DIR}")
    print("  Next step        : python 3_evaluate.py\n")


if __name__ == "__main__":
    main()
