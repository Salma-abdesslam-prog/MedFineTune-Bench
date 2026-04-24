"""
1_prepare_data.py — Download, clean, format, and split the medical Q&A dataset.

Output:
    data/train.jsonl
    data/val.jsonl
    data/test.jsonl

Each line is a JSON object:
    {"instruction": "<question>", "output": "<answer>"}
"""

import json
import os
import random
import sys

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Import central configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR,
    DATASET_SOURCES,
    MAX_EXAMPLES,
    MAX_OUTPUT_LENGTH,
    MIN_OUTPUT_LENGTH,
    SEED,
    SYSTEM_PROMPT,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

# ── Reproducibility ──────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────

def load_raw_dataset():
    """Try each dataset source in order and return the first one that loads."""
    for source in DATASET_SOURCES:
        try:
            print(f"[INFO] Trying dataset: {source}")
            ds = load_dataset(source)
            print(f"[OK]   Loaded '{source}' successfully.")
            return ds, source
        except Exception as e:
            print(f"[WARN] Could not load '{source}': {e}")
    raise RuntimeError(
        "All dataset sources failed. Check your internet connection or Hugging Face token."
    )


def extract_qa_pairs(ds, source_name: str) -> list[dict]:
    """
    Normalize different dataset schemas into a unified list of
    {"instruction": str, "output": str} dicts.
    """
    pairs = []

    # Flatten all available splits into one list
    all_splits = list(ds.keys())
    print(f"[INFO] Available splits: {all_splits}")

    for split in all_splits:
        data = ds[split]
        columns = data.column_names
        print(f"[INFO] Split '{split}' — columns: {columns} — rows: {len(data)}")

        for row in data:
            question, answer = None, None

            # --- lavita/MedQuAD ---
            if "question" in columns and "answer" in columns:
                question = row.get("question")
                answer   = row.get("answer")

            # --- keivalya/MedQuad-MedicalQnADataset ---
            elif "Question" in columns and "Answer" in columns:
                question = row.get("Question")
                answer   = row.get("Answer")

            # --- medalpaca/medical_meadow_medqa ---
            elif "input" in columns and "output" in columns:
                question = row.get("input")
                answer   = row.get("output")

            # Generic fallback: any instruction/output style
            elif "instruction" in columns and "output" in columns:
                question = row.get("instruction")
                answer   = row.get("output")

            if question and answer:
                pairs.append({"instruction": str(question).strip(),
                               "output": str(answer).strip()})

    return pairs


def clean_pairs(pairs: list[dict]) -> list[dict]:
    """Remove empty entries, duplicates, and length outliers."""
    cleaned = []
    seen_questions = set()

    for item in pairs:
        q = item["instruction"]
        a = item["output"]

        # Skip empty fields
        if not q or not a:
            continue

        # Skip very short or very long answers
        if len(a) < MIN_OUTPUT_LENGTH or len(a) > MAX_OUTPUT_LENGTH:
            continue

        # Deduplicate on question text
        q_lower = q.lower().strip()
        if q_lower in seen_questions:
            continue
        seen_questions.add(q_lower)

        cleaned.append({"instruction": q, "output": a})

    return cleaned


def format_as_qwen_chat(item: dict) -> dict:
    """
    Wrap instruction/output in the Qwen 2.5 chat format.
    The tokenizer's apply_chat_template will re-process this during training,
    but we store the raw text here for readability and compatibility.
    """
    return {
        "instruction": item["instruction"],
        "output": item["output"],
        "system": SYSTEM_PROMPT,
    }


def save_jsonl(data: list[dict], path: str):
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVED] {path}  ({len(data):,} examples)")


def print_stats(train, val, test):
    """Print dataset statistics after splitting."""
    all_data = train + val + test

    def avg_len(subset, key):
        return sum(len(x[key]) for x in subset) / max(len(subset), 1)

    print("\n" + "=" * 55)
    print("  DATASET STATISTICS")
    print("=" * 55)
    print(f"  Total examples  : {len(all_data):,}")
    print(f"  Train           : {len(train):,}")
    print(f"  Validation      : {len(val):,}")
    print(f"  Test            : {len(test):,}")
    print(f"  Avg question len: {avg_len(all_data, 'instruction'):.0f} chars")
    print(f"  Avg answer len  : {avg_len(all_data, 'output'):.0f} chars")
    print("=" * 55 + "\n")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n=== Step 1: Data Preparation ===\n")

    # 1. Load raw dataset
    ds, source = load_raw_dataset()

    # 2. Extract Q&A pairs from all splits
    pairs = extract_qa_pairs(ds, source)
    print(f"[INFO] Extracted {len(pairs):,} raw Q&A pairs.")

    if not pairs:
        print("[ERROR] No Q&A pairs extracted. Check dataset schema.")
        sys.exit(1)

    # 3. Clean
    pairs = clean_pairs(pairs)
    print(f"[INFO] After cleaning: {len(pairs):,} pairs.")

    # 4. Cap at MAX_EXAMPLES
    if len(pairs) > MAX_EXAMPLES:
        random.shuffle(pairs)
        pairs = pairs[:MAX_EXAMPLES]
        print(f"[INFO] Capped to {MAX_EXAMPLES:,} examples.")

    # 5. Format with Qwen chat template fields
    pairs = [format_as_qwen_chat(p) for p in tqdm(pairs, desc="Formatting")]

    # 6. Shuffle and split
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_data = pairs[:n_train]
    val_data   = pairs[n_train : n_train + n_val]
    test_data  = pairs[n_train + n_val :]

    # 7. Save splits
    os.makedirs(DATA_DIR, exist_ok=True)
    from config import TEST_FILE, TRAIN_FILE, VAL_FILE
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(val_data,   VAL_FILE)
    save_jsonl(test_data,  TEST_FILE)

    # 8. Print statistics
    print_stats(train_data, val_data, test_data)
    print("[DONE] Data preparation complete.\n")


if __name__ == "__main__":
    main()
