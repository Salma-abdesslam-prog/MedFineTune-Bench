"""
3_evaluate.py — Compare base model vs fine-tuned model on the test set.

Outputs:
    outputs/results.json  — structured metrics + per-example data

Metrics computed:
  - BLEU-4, ROUGE-L, BERTScore F1 (vs reference answer)
  - Average generation time (seconds)
  - Average perplexity on the test set
"""

import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    MAX_EVAL_EXAMPLES,
    OUTPUTS_DIR,
    RESULTS_FILE,
    SEED,
    SYSTEM_PROMPT,
    TEST_FILE,
)
from utils.inference import load_models
from utils.metrics import compute_bertscore, compute_bleu, compute_perplexity, compute_rouge_l

# Override max_new_tokens for eval — 128 tokens is enough for BLEU/ROUGE/BERTScore
# and makes evaluation 4x faster than the default 512
import utils.inference as _inf_module
import config as _cfg
_EVAL_MAX_NEW_TOKENS = 128

# ── Helpers ───────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test file not found: {path}\n"
            "Run 1_prepare_data.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_mean(lst: list) -> float:
    return round(float(np.mean(lst)), 4) if lst else 0.0


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n=== Step 3: Evaluation ===\n")

    # Ensure output directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load test set
    test_data = load_jsonl(TEST_FILE)
    if len(test_data) > MAX_EVAL_EXAMPLES:
        # Use a fixed slice for reproducibility
        test_data = test_data[:MAX_EVAL_EXAMPLES]
    print(f"[INFO] Evaluating on {len(test_data)} test examples.")

    # 2. Load models — temporarily lower max_new_tokens for faster eval
    _cfg.MAX_NEW_TOKENS = _EVAL_MAX_NEW_TOKENS
    _inf_module.MAX_NEW_TOKENS = _EVAL_MAX_NEW_TOKENS
    print(f"\n[INFO] Loading models (eval max_new_tokens={_EVAL_MAX_NEW_TOKENS})...")
    generate_original, generate_finetuned = load_models()

    # 3. Per-example generation and metric collection
    questions        = []
    references       = []
    original_answers = []
    finetuned_answers= []

    orig_bleu   = []; orig_rouge  = []; orig_times  = []
    ft_bleu     = []; ft_rouge    = []; ft_times    = []

    print("\n[INFO] Generating responses...\n")
    for item in tqdm(test_data, desc="Evaluating"):
        question  = item["instruction"]
        reference = item["output"]

        # Generate from both models
        orig_resp, orig_t = generate_original(question)
        ft_resp,   ft_t   = generate_finetuned(question)

        # Per-sample metrics
        orig_bleu.append(compute_bleu(orig_resp, reference))
        orig_rouge.append(compute_rouge_l(orig_resp, reference))
        orig_times.append(orig_t)

        ft_bleu.append(compute_bleu(ft_resp, reference))
        ft_rouge.append(compute_rouge_l(ft_resp, reference))
        ft_times.append(ft_t)

        questions.append(question)
        references.append(reference)
        original_answers.append(orig_resp)
        finetuned_answers.append(ft_resp)

    # 4. BERTScore (batched — more efficient)
    print("\n[INFO] Computing BERTScore (this takes a few minutes)...")
    orig_bert = compute_bertscore(original_answers, references)
    ft_bert   = compute_bertscore(finetuned_answers, references)

    # 5. Perplexity — reuse the already-loaded model to avoid double VRAM usage
    print("[INFO] Computing perplexity (reusing loaded model)...")
    import utils.inference as _inf
    _tokenizer = _inf._tokenizer
    _model_ref  = _inf._model

    # Base model: disable the LoRA adapter
    if _inf._is_peft:
        with _model_ref.disable_adapter():
            orig_ppl = compute_perplexity(_model_ref, _tokenizer, references[:20])
    else:
        orig_ppl = compute_perplexity(_model_ref, _tokenizer, references[:20])

    # Fine-tuned model: adapter enabled (default state)
    ft_ppl = compute_perplexity(_model_ref, _tokenizer, references[:20])
    torch.cuda.empty_cache()

    # 6. Build results structure
    examples = []
    for i in range(len(questions)):
        examples.append({
            "question":         questions[i],
            "reference":        references[i],
            "original_answer":  original_answers[i],
            "finetuned_answer": finetuned_answers[i],
            "metrics": {
                "original_bleu":        orig_bleu[i],
                "original_rouge_l":     orig_rouge[i],
                "original_bertscore":   orig_bert[i],
                "original_time_s":      round(orig_times[i], 3),
                "finetuned_bleu":       ft_bleu[i],
                "finetuned_rouge_l":    ft_rouge[i],
                "finetuned_bertscore":  ft_bert[i],
                "finetuned_time_s":     round(ft_times[i], 3),
            },
        })

    results = {
        "original": {
            "bleu":         safe_mean(orig_bleu),
            "rouge_l":      safe_mean(orig_rouge),
            "bertscore_f1": safe_mean(orig_bert),
            "perplexity":   orig_ppl,
            "avg_time_s":   safe_mean(orig_times),
        },
        "finetuned": {
            "bleu":         safe_mean(ft_bleu),
            "rouge_l":      safe_mean(ft_rouge),
            "bertscore_f1": safe_mean(ft_bert),
            "perplexity":   ft_ppl,
            "avg_time_s":   safe_mean(ft_times),
        },
        "examples": examples,
        "meta": {
            "num_examples": len(examples),
            "adapter_dir":  ADAPTER_DIR,
            "base_model":   BASE_MODEL_NAME,
        },
    }

    # 7. Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {RESULTS_FILE}")

    # 8. Print summary
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<20} {'Original':>10} {'Fine-tuned':>10}")
    print("-" * 55)
    for key in ["bleu", "rouge_l", "bertscore_f1", "perplexity", "avg_time_s"]:
        o = results["original"][key]
        f = results["finetuned"][key]
        marker = "  <<" if (
            (key != "perplexity" and key != "avg_time_s" and f > o) or
            (key == "perplexity" and f < o)
        ) else ""
        print(f"  {key:<20} {str(o):>10} {str(f):>10}{marker}")
    print("=" * 55)
    print("\n[DONE] Evaluation complete.\n")
    print("  Next step: python 4_app.py\n")


if __name__ == "__main__":
    main()
