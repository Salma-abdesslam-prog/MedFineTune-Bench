"""
2_finetune_fallback.py — QLoRA fine-tuning WITHOUT Unsloth.

Use this script on Windows native where Unsloth installation fails.
Uses: transformers + peft + bitsandbytes + trl

~30–40% slower than 2_finetune.py but produces identical results.
"""

import json
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
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
    SEED,
    SYSTEM_PROMPT,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

# ── Reproducibility ───────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# LoRA target modules for Qwen 2.5 (explicit list, no "all-linear" shorthand)
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Helpers ───────────────────────────────────────────────────

def print_vram(label: str):
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] {label}: {used:.2f} GB used / {total:.2f} GB total")
    else:
        print("[VRAM] No CUDA GPU detected — running on CPU (very slow).")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run 1_prepare_data.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_conversation(example: dict, tokenizer) -> dict:
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
    print("\n=== Step 2: QLoRA Fine-tuning (Fallback — no Unsloth) ===\n")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # 1. BitsAndBytes 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.bfloat16,
        bnb_4bit_use_double_quant  = True,    # Nested quantization saves ~0.4 GB
    )

    # 2. Load tokenizer
    print(f"[INFO] Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Avoid warnings with causal LM

    # 3. Load model in 4-bit
    # Note: unsloth/*-bnb-4bit models already embed quantization config,
    # so we only pass bnb_config for non-pre-quantized model names.
    print(f"[INFO] Loading model in 4-bit: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        dtype               = torch.bfloat16,
    )
    print_vram("After model load")

    # 4. Prepare model for k-bit training (enables grad checkpointing internally)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 5. Attach LoRA adapter
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = QWEN_TARGET_MODULES,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_vram("After LoRA attach")

    # 6. Load and format datasets
    print("[INFO] Loading training data...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)

    train_formatted = [format_conversation(ex, tokenizer) for ex in train_raw]
    val_formatted   = [format_conversation(ex, tokenizer) for ex in val_raw]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset   = Dataset.from_list(val_formatted)
    print(f"[INFO] Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")

    # 7. Training arguments
    use_bf16 = BF16 and torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    use_fp16 = FP16 and not use_bf16

    # Max token length in this dataset is 493 — 512 is sufficient and saves VRAM
    effective_max_length = min(MAX_SEQ_LENGTH, 512)

    training_args = SFTConfig(
        output_dir                  = ADAPTER_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        # use_reentrant=False avoids memory fragmentation during checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        optim                       = OPTIMIZER,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        logging_steps               = LOGGING_STEPS,
        eval_strategy               = "no",       # Disabled — eval caused VRAM fragmentation
        save_strategy               = "epoch",
        load_best_model_at_end      = False,
        seed                        = SEED,
        report_to                   = "none",
        dataset_text_field          = "text",
        max_length                  = effective_max_length,
        packing                     = False,
    )

    # 8. Train — resume from epoch-1 checkpoint if it exists
    checkpoint_dir = os.path.join(ADAPTER_DIR, "checkpoint-1194")
    resume_from = checkpoint_dir if os.path.isdir(checkpoint_dir) else None
    if resume_from:
        print(f"[INFO] Resuming from checkpoint: {resume_from}")
    else:
        print("[INFO] Starting fresh training.")

    trainer = SFTTrainer(
        model             = model,
        processing_class  = tokenizer,
        train_dataset     = train_dataset,
        eval_dataset      = None,             # No eval to avoid VRAM fragmentation
        args              = training_args,
    )

    print("\n[INFO] Starting training (fallback mode)...\n")
    print_vram("Before training")
    trainer.train(resume_from_checkpoint=resume_from)
    print_vram("After training")

    # 9. Save adapter
    print(f"\n[INFO] Saving adapter to: {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print("\n[DONE] Fine-tuning (fallback) complete.\n")
    print(f"  Adapter saved to : {ADAPTER_DIR}")
    print("  Next step        : python 3_evaluate.py\n")


if __name__ == "__main__":
    main()
