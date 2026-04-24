"""
utils/inference.py — Model loading and text generation utilities.

Loads the base model ONCE in 4-bit, then exposes two generation functions
that switch the LoRA adapter on/off without reloading weights.
"""

import os
import sys
import time
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    SEED,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)

# Module-level cache so the Gradio app only loads once
_model     = None
_tokenizer = None
_is_peft   = False


def load_models():
    """
    Load the base model in 4-bit quantization and attach the LoRA adapter.

    Returns
    -------
    generate_original : callable
        Generate text with the adapter DISABLED (base model behavior).
    generate_finetuned : callable
        Generate text with the adapter ENABLED (fine-tuned behavior).
    """
    global _model, _tokenizer, _is_peft

    if _model is not None:
        # Already loaded — return the two wrappers directly
        return _make_generate_original(), _make_generate_finetuned()

    # 1. Tokenizer
    print(f"[INFO] Loading tokenizer from: {ADAPTER_DIR}")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    except Exception:
        print(f"[WARN] Tokenizer not found in {ADAPTER_DIR}, falling back to base model.")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    _tokenizer.pad_token = _tokenizer.eos_token

    # 2. BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    # 3. Base model
    print(f"[INFO] Loading base model: {BASE_MODEL_NAME}")
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        torch_dtype         = torch.bfloat16,
    )

    # 4. Attach LoRA adapter if it exists
    if os.path.exists(ADAPTER_DIR) and os.path.exists(
        os.path.join(ADAPTER_DIR, "adapter_config.json")
    ):
        print(f"[INFO] Loading LoRA adapter from: {ADAPTER_DIR}")
        _model = PeftModel.from_pretrained(
            _model, ADAPTER_DIR, is_trainable=False
        )
        _is_peft = True
        print("[INFO] LoRA adapter loaded successfully.")
    else:
        print(f"[WARN] No adapter found at {ADAPTER_DIR}.")
        print("[WARN] Both 'original' and 'finetuned' will use the base model.")
        _is_peft = False

    _model.eval()
    print("[INFO] Models ready.")

    return _make_generate_original(), _make_generate_finetuned()


# ── Private generation helpers ────────────────────────────────

def _build_prompt(user_message: str) -> str:
    """Build a formatted chat prompt using the Qwen 2.5 template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _generate(prompt: str) -> tuple[str, float]:
    """
    Run greedy/sampling generation and return (response_text, elapsed_seconds).
    The model and adapter state are controlled by the caller.
    """
    inputs = _tokenizer(
        prompt,
        return_tensors   = "pt",
        truncation       = True,
        max_length       = 1536,   # Reserve tokens for the response
        padding          = False,
    ).to(_model.device)

    t0 = time.time()
    # Read MAX_NEW_TOKENS at call time so callers can override it via the module var
    import utils.inference as _self
    max_new_tokens = getattr(_self, 'MAX_NEW_TOKENS', MAX_NEW_TOKENS)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens      = max_new_tokens,
            temperature         = TEMPERATURE,
            top_p               = TOP_P,
            do_sample           = DO_SAMPLE,
            repetition_penalty  = REPETITION_PENALTY,
            pad_token_id        = _tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response   = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response, elapsed


def _make_generate_original():
    """Return a callable that generates text with the adapter disabled."""
    def generate_original(user_message: str) -> tuple[str, float]:
        prompt = _build_prompt(user_message)
        if _is_peft:
            with _model.disable_adapter():
                return _generate(prompt)
        return _generate(prompt)
    return generate_original


def _make_generate_finetuned():
    """Return a callable that generates text with the adapter enabled."""
    def generate_finetuned(user_message: str) -> tuple[str, float]:
        prompt = _build_prompt(user_message)
        # Adapter is enabled by default in a PeftModel
        return _generate(prompt)
    return generate_finetuned
