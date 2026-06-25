"""
4_app.py — Gradio interface: side-by-side comparison of original vs fine-tuned model.

Two tabs:
  1. Comparative Chat   — live Q&A with generation time
  2. Evaluation Metrics — charts and examples from results.json
"""

import json
import os
import sys
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_FILE

# ── Model loading (deferred until first query) ────────────────
_generate_original  = None
_generate_finetuned = None

def _ensure_models_loaded():
    global _generate_original, _generate_finetuned
    if _generate_original is None:
        from utils.inference import load_models
        _generate_original, _generate_finetuned = load_models()


# ── Chat logic ────────────────────────────────────────────────

def answer_question(question: str):
    """
    Generate answers from both models and yield incremental UI updates.
    Yields a tuple: (orig_text, orig_time_label, ft_text, ft_time_label)
    """
    if not question or not question.strip():
        yield "Please enter a question.", "", "Please enter a question.", ""
        return

    _ensure_models_loaded()

    # Show loading state immediately
    yield "⏳ Generating...", "", "⏳ Generating...", ""

    # Original model
    t0 = time.time()
    orig_resp, orig_t = _generate_original(question.strip())
    yield orig_resp, f"⏱ {orig_t:.2f}s", "⏳ Generating...", ""

    # Fine-tuned model
    ft_resp, ft_t = _generate_finetuned(question.strip())
    yield orig_resp, f"⏱ {orig_t:.2f}s", ft_resp, f"⏱ {ft_t:.2f}s"


# ── Metrics loading ───────────────────────────────────────────

def load_results() -> dict | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics_table(results: dict) -> str:
    """Build an HTML table with metric comparison, highlighting best values."""
    orig = results["original"]
    ft   = results["finetuned"]

    metric_labels = {
        "cosine_similarity":  ("Cosine Similarity ↑",  True),
        "medical_accuracy":   ("Medical Accuracy ↑",   True),
        "hallucination_rate": ("Hallucination Rate ↓",  False),
        "avg_time_s":         ("Avg Time (s) ↓",        False),
    }

    rows = ""
    for key, (label, higher_is_better) in metric_labels.items():
        o_val = orig.get(key) or 0.0
        f_val = ft.get(key)   or 0.0

        if higher_is_better:
            o_style = ' style="color:green;font-weight:bold"' if o_val > f_val else ""
            f_style = ' style="color:green;font-weight:bold"' if f_val > o_val else ""
        else:
            o_style = ' style="color:green;font-weight:bold"' if o_val < f_val else ""
            f_style = ' style="color:green;font-weight:bold"' if f_val < o_val else ""

        rows += (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td{o_style}>{o_val:.4f}</td>"
            f"<td{f_style}>{f_val:.4f}</td>"
            f"</tr>"
        )

    table = f"""
    <table style="width:100%;border-collapse:collapse;font-size:15px">
      <thead>
        <tr style="background:#f0f0f0">
          <th style="padding:8px;text-align:left">Metric</th>
          <th style="padding:8px;text-align:center">Original</th>
          <th style="padding:8px;text-align:center">Fine-tuned</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <p style="font-size:12px;color:#666">
      ↑ Higher is better &nbsp;|&nbsp; ↓ Lower is better &nbsp;|&nbsp;
      <span style="color:green;font-weight:bold">Green</span> = best value
    </p>
    """
    return table


def build_bar_chart(results: dict):
    """Return a matplotlib Figure comparing medical evaluation metrics (all normalised to 0–1)."""
    orig = results["original"]
    ft   = results["finetuned"]

    metrics = ["Cosine\nSimilarity", "Medical\nAccuracy"]
    orig_vals = [
        orig.get("cosine_similarity", 0),
        orig.get("medical_accuracy",  0) / 10,
    ]
    ft_vals = [
        ft.get("cosine_similarity", 0),
        ft.get("medical_accuracy",  0) / 10,
    ]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - w / 2, orig_vals, w, label="Original",   color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w / 2, ft_vals,   w, label="Fine-tuned", color="#DD8452", alpha=0.85)

    ax.set_ylabel("Score (0–1, normalised)")
    ax.set_title("Original vs Fine-tuned — Medical Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    return fig


def build_examples_html(results: dict, n: int = 5) -> str:
    """Select the n examples with the largest cosine similarity difference."""
    examples = results.get("examples", [])
    if not examples:
        return "<p>No examples available.</p>"

    def score_diff(ex):
        m = ex.get("metrics", {})
        ft_cos   = m.get("finetuned", {}).get("cosine_similarity", 0)
        orig_cos = m.get("original",  {}).get("cosine_similarity", 0)
        return abs(ft_cos - orig_cos)

    top = sorted(examples, key=score_diff, reverse=True)[:n]

    html = ""
    for i, ex in enumerate(top, 1):
        m    = ex.get("metrics", {})
        orig = m.get("original",  {})
        ft   = m.get("finetuned", {})
        diff = ft.get("cosine_similarity", 0) - orig.get("cosine_similarity", 0)
        direction = "🟢 Fine-tuned better" if diff > 0 else "🔴 Original better"

        html += f"""
        <details style="margin-bottom:12px;border:1px solid #ddd;border-radius:6px;padding:8px">
          <summary style="font-weight:bold;cursor:pointer">
            Example {i} — {direction}
            (Cosine Similarity Δ = {diff:+.3f})
          </summary>
          <p><strong>Question:</strong> {ex['question']}</p>
          <p><strong>Reference:</strong> {ex['reference'][:300]}{'...' if len(ex['reference']) > 300 else ''}</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
            <div style="background:#f5f5ff;padding:8px;border-radius:4px">
              <strong>Original</strong>
              <p style="font-size:13px">{ex['original_answer'][:300]}{'...' if len(ex['original_answer']) > 300 else ''}</p>
              <small>Cosine: {orig.get('cosine_similarity', 0):.3f} | Accuracy: {orig.get('medical_accuracy', 0):.1f} | Hall.: {orig.get('hallucination_rate', 0):.3f}</small>
            </div>
            <div style="background:#f5fff5;padding:8px;border-radius:4px">
              <strong>Fine-tuned</strong>
              <p style="font-size:13px">{ex['finetuned_answer'][:300]}{'...' if len(ex['finetuned_answer']) > 300 else ''}</p>
              <small>Cosine: {ft.get('cosine_similarity', 0):.3f} | Accuracy: {ft.get('medical_accuracy', 0):.1f} | Hall.: {ft.get('hallucination_rate', 0):.3f}</small>
            </div>
          </div>
        </details>
        """
    return html


# ── Gradio UI construction ────────────────────────────────────

APP_CSS = """
    .disclaimer { color: #c0392b; font-weight: bold; text-align: center;
                  padding: 10px; border: 2px solid #e74c3c;
                  border-radius: 6px; background: #fdf2f2; }
    .model-label { font-size: 16px; font-weight: bold; text-align: center; }
"""

DISCLAIMER = (
    "⚠️ This tool is for educational purposes only and does not replace "
    "professional medical advice. Always consult a qualified healthcare professional."
)

EXAMPLE_QUESTIONS = [
    "What are the symptoms of type 2 diabetes?",
    "How is hypertension treated?",
    "What is the difference between a cold and the flu?",
    "What are the side effects of ibuprofen?",
]

def create_interface() -> gr.Blocks:
    results = load_results()

    with gr.Blocks(title="Medical Chatbot — Original vs Fine-tuned") as demo:

        # ── Header ──────────────────────────────────────────────
        gr.Markdown(
            "# 🏥 Medical Chatbot — Original vs Fine-tuned Comparison\n"
            "Compare responses from the base **Qwen 2.5 3B** model and the "
            "**QLoRA fine-tuned** version on your medical questions."
        )
        gr.HTML(f'<div class="disclaimer">{DISCLAIMER}</div>')

        # ── Tab 1: Comparative Chat ──────────────────────────────
        with gr.Tab("💬 Comparative Chat"):
            with gr.Row():
                question_box = gr.Textbox(
                    label       = "Your medical question",
                    placeholder = "e.g. What are the symptoms of type 2 diabetes?",
                    lines       = 3,
                    scale       = 5,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="model-label">📦 Original Model</div>')
                    orig_output = gr.Textbox(
                        label    = "Response",
                        lines    = 10,
                        max_lines= 20,
                        interactive = False,
                    )
                    orig_time = gr.Markdown("")

                with gr.Column():
                    gr.HTML('<div class="model-label">🎯 Fine-tuned Model</div>')
                    ft_output = gr.Textbox(
                        label    = "Response",
                        lines    = 10,
                        max_lines= 20,
                        interactive = False,
                    )
                    ft_time = gr.Markdown("")

            # Pre-filled example buttons
            gr.Markdown("**Quick examples — click to fill the question box:**")
            with gr.Row():
                for ex in EXAMPLE_QUESTIONS:
                    gr.Button(ex[:50] + "…" if len(ex) > 50 else ex, size="sm").click(
                        fn      = lambda q=ex: q,
                        outputs = question_box,
                    )

            # Wire up the Send button
            send_btn.click(
                fn      = answer_question,
                inputs  = question_box,
                outputs = [orig_output, orig_time, ft_output, ft_time],
            )
            question_box.submit(
                fn      = answer_question,
                inputs  = question_box,
                outputs = [orig_output, orig_time, ft_output, ft_time],
            )

            gr.HTML(f'<div class="disclaimer" style="margin-top:16px">{DISCLAIMER}</div>')

        # ── Tab 2: Evaluation Metrics ────────────────────────────
        with gr.Tab("📊 Evaluation Metrics"):
            if results is None:
                gr.Markdown(
                    "⚠️ **No results found.**\n\n"
                    f"Run `python 3_evaluate.py` first to generate `{RESULTS_FILE}`."
                )
            else:
                meta = results.get("meta", {})
                gr.Markdown(f"Base model: `{meta.get('base_model', 'unknown')}`")

                # Metrics table
                gr.Markdown("### 📋 Aggregate Metrics")
                gr.HTML(build_metrics_table(results))

                # Bar chart
                gr.Markdown("### 📈 Visual Comparison")
                gr.Plot(value=build_bar_chart(results))

                # Hallucination rate note
                orig_hr = results["original"].get("hallucination_rate") or 0.0
                ft_hr   = results["finetuned"].get("hallucination_rate") or 0.0
                better  = "Fine-tuned" if ft_hr < orig_hr else "Original"
                gr.Markdown(
                    f"**Hallucination Rate** — Original: `{orig_hr:.3f}` | "
                    f"Fine-tuned: `{ft_hr:.3f}` | "
                    f"Better: **{better}** (lower = better)"
                )

                # Notable examples
                gr.Markdown("### 🔍 Notable Examples (largest cosine similarity difference)")
                gr.HTML(build_examples_html(results, n=5))

    return demo


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    print("[INFO] Building Gradio interface...")
    demo = create_interface()
    print("[INFO] Starting server at http://localhost:7860")
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        show_error  = True,
        theme       = gr.themes.Soft(),
        css         = APP_CSS,
    )
