# 🏥 Medical Chatbot — QLoRA Fine-tuned Qwen 2.5 3B

End-to-end NLP project: dataset preparation → QLoRA fine-tuning → quantitative evaluation → production web interface comparing the base and fine-tuned models side-by-side.

## Results

Evaluated on 22 test examples from MedQuAD. Cosine Similarity uses PubMedBERT embeddings; Medical Accuracy and Hallucination Rate use heuristic fallbacks (set `JUDGE_API_KEY` for LLM-as-a-Judge scoring).

| Metric | Base Model | Fine-tuned | Δ |
|--------|-----------|------------|---|
| Cosine Similarity ↑ | 0.9686 | **0.9712** | +0.0026 |
| Medical Accuracy ↑ (0–10) | 2.9561 | **3.3784** | +0.4223 |
| Hallucination Rate ↓ (0–1) | 0.3787 | **0.1330** | −0.2457 |
| Avg Inference Time (s) | **31.60** | 37.90 | +6.30 |

---

## Stack

- **Model**: [Qwen 2.5 3B Instruct](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-bnb-4bit) (4-bit NF4 quantized)
- **Fine-tuning**: QLoRA via PEFT + TRL `SFTTrainer` (r=16, α=16, target all-linear layers)
- **Dataset**: [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) — 10 610 medical Q&A pairs
- **Backend**: FastAPI + Server-Sent Events (SSE) streaming
- **Frontend**: Custom web UI — HTML/CSS/JS, no framework dependencies
- **Metrics**: Cosine Semantic Similarity, Medical Accuracy (LLM-as-a-Judge), Hallucination Rate

---

## Project Structure

```
medical_chatbot/
├── config.py                 # All hyperparameters in one place
├── 1_prepare_data.py         # Download MedQuAD, clean, 90/5/5 split
├── 2_finetune_fallback.py    # QLoRA training (Windows-compatible)
├── 2_finetune.py             # QLoRA training (Unsloth — Linux/WSL2 only)
├── 3_evaluate.py             # Cosine Similarity / Medical Accuracy / Hallucination Rate
├── api.py                    # FastAPI backend with SSE streaming  ← main entry point
├── frontend/
│   └── index.html            # Chat UI (side-by-side comparison, metrics view)
├── outputs/
│   ├── lora_adapter/         # adapter_config.json + tokenizer (weights excluded)
│   └── results.json          # Evaluation results
├── utils/
│   ├── inference.py          # Model loading, adapter toggling, generation
│   └── metrics.py            # Medical metric helpers (embeddings + LLM judge)
└── requirements.txt
```

> **Note**: `adapter_model.safetensors` (115 MB) and training checkpoints are excluded from this repo.
> The adapter is published on Hugging Face Hub: *(link if you upload it)*

---

## Hardware Used

| | |
|---|---|
| GPU | NVIDIA RTX A1000 6 GB Laptop GPU |
| Training time | ~6 h (2 epochs, 9 549 examples, batch size 1 + grad accum 8) |
| Inference | ~40–60 s/query (4-bit quantized, CPU tokenizer) |

---

## Quick Start

### 1. Install dependencies
```bash
# PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python 1_prepare_data.py       # Download + split dataset (~5 min)
python 2_finetune_fallback.py  # Train QLoRA adapter (~4–6 h on 6 GB GPU)
python 3_evaluate.py           # Evaluate on test examples (~25 min)
python api.py                  # Start web UI → http://localhost:8000
```

> **Windows users**: use `2_finetune_fallback.py`.  
> **Linux / WSL2**: install Unsloth first and use `2_finetune.py` for ~30% faster training.
>
> `4_app.py` is an older Gradio prototype — use `api.py` for the full interface.

---

## Key Design Decisions

**Single model load, adapter toggling**: both "Original" and "Fine-tuned" inference use the same loaded weights. The LoRA adapter is toggled on/off with `model.disable_adapter()`, saving ~2 GB VRAM vs loading two separate models.

**No mid-training evaluation**: disabling `eval_strategy` during fine-tuning prevents VRAM fragmentation between epochs — a common cause of slowdowns on 6 GB cards.

**Reduced sequence length**: analysis of the dataset showed 99th-percentile token length of 493. Setting `max_length=512` instead of 2048 reduced VRAM usage by ~4× with no quality impact.

---

## Interface

The web UI (`api.py` + `frontend/index.html`) features:
- Side-by-side **Compare mode** — both models answer the same question simultaneously
- **Metrics dashboard** — Chart.js bar chart + per-example cosine similarity breakdown
- SSE streaming — typing indicator while the model generates
- No framework dependencies — pure HTML/CSS/JS

---

## Disclaimer

> ⚠️ For **educational and research purposes only**. Model outputs do not constitute medical advice. Always consult a qualified healthcare professional.
