"""
3_evaluate.py — Compare base model vs fine-tuned model on the test set.

Outputs:
    outputs/results.json  — structured metrics + per-example data

Metrics:
  - Cosine semantic similarity  (biomedical sentence embeddings, 0–1)
  - Medical Accuracy Score      (LLM-as-a-Judge, 0–10)
  - Hallucination Rate          (claim-level classification, 0–1)

LLM judge (optional): set JUDGE_API_KEY or OPENAI_API_KEY environment variable.
  Without a key, LLM-judge metrics fall back to heuristic approximations.
"""

import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    MAX_EVAL_EXAMPLES,
    OUTPUTS_DIR,
    RESULTS_FILE,
    TEST_FILE,
)
from utils.inference import load_models
from utils.metrics import (
    compute_semantic_similarity_batch,
    compute_medical_accuracy,
    compute_hallucination_rate,
)

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

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load test set
    test_data = load_jsonl(TEST_FILE)
    if len(test_data) > MAX_EVAL_EXAMPLES:
        test_data = test_data[:MAX_EVAL_EXAMPLES]
    print(f"[INFO] Evaluating on {len(test_data)} test examples.")

    # 2. Load models
    _cfg.MAX_NEW_TOKENS        = _EVAL_MAX_NEW_TOKENS
    _inf_module.MAX_NEW_TOKENS = _EVAL_MAX_NEW_TOKENS
    print(f"\n[INFO] Loading models (eval max_new_tokens={_EVAL_MAX_NEW_TOKENS})...")
    generate_original, generate_finetuned = load_models()

    # 3. Generate responses from both models
    questions         = []
    references        = []
    original_answers  = []
    finetuned_answers = []
    orig_times        = []
    ft_times          = []

    print("\n[INFO] Generating responses...\n")
    for item in tqdm(test_data, desc="Generating"):
        question  = item["instruction"]
        reference = item["output"]

        orig_resp, orig_t = generate_original(question)
        ft_resp,   ft_t   = generate_finetuned(question)

        questions.append(question)
        references.append(reference)
        original_answers.append(orig_resp)
        finetuned_answers.append(ft_resp)
        orig_times.append(orig_t)
        ft_times.append(ft_t)

    # 4. Semantic similarity (batched for efficiency)
    print("\n[INFO] Computing semantic similarity (biomedical embeddings)...")
    orig_cosine = compute_semantic_similarity_batch(original_answers, references)
    ft_cosine   = compute_semantic_similarity_batch(finetuned_answers, references)

    # 5. LLM-as-a-Judge metrics (per example)
    print("\n[INFO] Computing LLM-as-a-Judge metrics...")
    orig_accuracy      = []
    orig_hallucination = []
    ft_accuracy        = []
    ft_hallucination   = []

    for i in tqdm(range(len(questions)), desc="LLM judging"):
        q = questions[i]
        r = references[i]
        o = original_answers[i]
        f = finetuned_answers[i]

        orig_accuracy.append(compute_medical_accuracy(q, o, r))
        orig_hallucination.append(compute_hallucination_rate(o, r))

        ft_accuracy.append(compute_medical_accuracy(q, f, r))
        ft_hallucination.append(compute_hallucination_rate(f, r))

    # 6. Build results structure
    examples = []
    for i in range(len(questions)):
        examples.append({
            "question":         questions[i],
            "reference":        references[i],
            "original_answer":  original_answers[i],
            "finetuned_answer": finetuned_answers[i],
            "metrics": {
                "original": {
                    "cosine_similarity":  orig_cosine[i],
                    "medical_accuracy":   orig_accuracy[i],
                    "hallucination_rate": orig_hallucination[i],
                    "time_s":             round(orig_times[i], 3),
                },
                "finetuned": {
                    "cosine_similarity":  ft_cosine[i],
                    "medical_accuracy":   ft_accuracy[i],
                    "hallucination_rate": ft_hallucination[i],
                    "time_s":             round(ft_times[i], 3),
                },
            },
        })

    results = {
        "original": {
            "cosine_similarity":  safe_mean(orig_cosine),
            "medical_accuracy":   safe_mean(orig_accuracy),
            "hallucination_rate": safe_mean(orig_hallucination),
            "avg_time_s":         safe_mean(orig_times),
        },
        "finetuned": {
            "cosine_similarity":  safe_mean(ft_cosine),
            "medical_accuracy":   safe_mean(ft_accuracy),
            "hallucination_rate": safe_mean(ft_hallucination),
            "avg_time_s":         safe_mean(ft_times),
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

    # 8. Print summary table
    _HIGHER_IS_BETTER = {"cosine_similarity", "medical_accuracy"}
    _LOWER_IS_BETTER  = {"hallucination_rate"}

    print("\n" + "=" * 62)
    print("  EVALUATION RESULTS")
    print("=" * 62)
    print(f"  {'Metric':<25} {'Original':>10} {'Fine-tuned':>10}")
    print("-" * 62)
    for key in ["cosine_similarity", "medical_accuracy", "hallucination_rate", "avg_time_s"]:
        o = results["original"][key]
        f = results["finetuned"][key]
        if key in _HIGHER_IS_BETTER:
            marker = "  <<" if f > o else ""
        elif key in _LOWER_IS_BETTER:
            marker = "  <<" if f < o else ""
        else:
            marker = ""
        print(f"  {key:<25} {str(o):>10} {str(f):>10}{marker}")
    print("=" * 62)
    print("\n  Note: << marks improvement in the fine-tuned model.")
    print("        hallucination_rate: lower is better.")
    print("\n[DONE] Evaluation complete.\n")
    print("  Next step: python 4_app.py\n")


if __name__ == "__main__":
    main()
