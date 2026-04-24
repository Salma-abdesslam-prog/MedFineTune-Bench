"""
utils/metrics.py — NLP evaluation metrics for the medical chatbot.

Metrics computed:
  - BLEU-4       (sacrebleu)
  - ROUGE-L      (rouge-score)
  - BERTScore F1 (bert-score, microsoft/deberta-xlarge-mnli)
  - Perplexity   (computed from model loss on reference tokens)
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports to avoid loading heavy models at module import time
_bertscore_scorer = None
_sacrebleu        = None


def compute_bleu(hypothesis: str, reference: str) -> float:
    """
    Compute corpus-level BLEU-4 score for a single hypothesis/reference pair.
    Returns a score in [0, 100].
    """
    global _sacrebleu
    if _sacrebleu is None:
        import sacrebleu as sb
        _sacrebleu = sb
    result = _sacrebleu.corpus_bleu([hypothesis], [[reference]])
    return round(result.score, 4)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score. Returns a score in [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores["rougeL"].fmeasure, 4)


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str = "distilbert-base-uncased",   # lightweight for 6 GB VRAM
    device: Optional[str] = None,
) -> list[float]:
    """
    Compute BERTScore F1 for a batch of hypothesis/reference pairs.
    Returns a list of F1 scores in [0, 1].

    Uses distilbert-base-uncased by default for speed and VRAM efficiency.
    For higher accuracy, switch to 'microsoft/deberta-xlarge-mnli'
    (requires ~3 GB extra VRAM — not recommended on 6 GB).
    """
    import bert_score

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score.score(
        hypotheses,
        references,
        model_type = model_type,
        lang       = lang,
        device     = device,
        verbose    = False,
    )
    return [round(f.item(), 4) for f in F1]


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
) -> float:
    """
    Compute average per-token perplexity over a list of text strings.

    Uses a sliding-window approach to handle sequences longer than max_length.
    Lower perplexity = better language model fit.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll  = 0.0
    total_toks = 0

    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors = "pt",
            truncation     = False,
        )
        seq_len = encodings.input_ids.shape[1]

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end      = min(begin + max_length, seq_len)
            # Only score the new tokens (avoid double-counting the overlap)
            tgt_len  = end - prev_end
            input_ids = encodings.input_ids[:, begin:end].to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                # loss is mean NLL over all tokens in the window
                nll = outputs.loss * input_ids.shape[1]

            total_nll  += nll.item()
            total_toks += tgt_len
            prev_end    = end

            if end == seq_len:
                break

    if total_toks == 0:
        return float("inf")

    avg_nll    = total_nll / total_toks
    perplexity = np.exp(avg_nll)
    return round(float(perplexity), 4)


def compute_all_metrics(
    hypotheses : list[str],
    references : list[str],
    model      = None,
    tokenizer  = None,
) -> dict:
    """
    Convenience function that computes BLEU, ROUGE-L, BERTScore, and perplexity
    for a list of hypothesis/reference pairs.

    Returns a dict with mean values:
        {
            "bleu":          float,  # 0–100
            "rouge_l":       float,  # 0–1
            "bertscore_f1":  float,  # 0–1
            "perplexity":    float,  # > 0, lower is better
        }
    """
    assert len(hypotheses) == len(references), "Mismatched hypothesis/reference counts."

    bleu_scores   = [compute_bleu(h, r)    for h, r in zip(hypotheses, references)]
    rouge_scores  = [compute_rouge_l(h, r) for h, r in zip(hypotheses, references)]
    bert_scores   = compute_bertscore(hypotheses, references)

    result = {
        "bleu":         round(float(np.mean(bleu_scores)),  4),
        "rouge_l":      round(float(np.mean(rouge_scores)), 4),
        "bertscore_f1": round(float(np.mean(bert_scores)),  4),
        "perplexity":   None,
    }

    if model is not None and tokenizer is not None:
        ppl = compute_perplexity(model, tokenizer, references)
        result["perplexity"] = ppl

    return result
