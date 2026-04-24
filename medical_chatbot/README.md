# 🏥 Medical Chatbot — Qwen 2.5 3B Fine-tuned with QLoRA

Complete pipeline: data preparation → fine-tuning → evaluation → comparative Gradio interface.

---

## Hardware Prerequisites

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA with ≥ 6 GB VRAM (RTX A1000 / RTX 3060 / etc.) |
| RAM | 16 GB recommended |
| Storage | ~15 GB free (model + data + adapter) |
| CUDA | 11.8 or 12.x |
| Driver | ≥ 525.x — verify with `nvidia-smi` |

---

## ⚠️ Unsloth on Windows

Unsloth is **optimized for Linux/WSL2**. Native Windows installation often fails.

### Option A — WSL2 (recommended)
```powershell
# In PowerShell (admin)
wsl --install
# Then follow Linux instructions inside the WSL2 terminal
```

### Option B — Native Windows fallback
If Unsloth fails to install, use `2_finetune_fallback.py` instead of `2_finetune.py`.
The fallback uses `transformers + peft + bitsandbytes + trl` directly — ~30–40% slower but fully functional.

---

## Installation

### 1. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / WSL2
source venv/bin/activate
```

### 2. Install PyTorch with CUDA support
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Unsloth (WSL2 / Linux only)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 5. (Windows native only) Install bitsandbytes wheel
```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

---

## Execution Order

```
Step 1:  python 1_prepare_data.py          (~5–10 min, download + processing)
Step 2a: python 2_finetune.py              (~2–4 h, requires Unsloth)
  OR
Step 2b: python 2_finetune_fallback.py     (~3–6 h, Windows-compatible fallback)
Step 3:  python 3_evaluate.py              (~15–30 min on 100 test examples)
Step 4:  python 4_app.py                   (starts Gradio server → http://localhost:7860)
```

---

## Time Estimates

| Step | Estimated Duration | Notes |
|------|--------------------|-------|
| Data preparation | 5–10 min | Depends on download speed |
| Fine-tuning (Unsloth) | 2–4 h | 2 epochs, 15k examples, RTX A1000 |
| Fine-tuning (fallback) | 3–6 h | Without Unsloth optimizations |
| Evaluation | 15–30 min | 100 test examples |
| Gradio interface | Immediate | Model loading ~1–2 min |

---

## Project Structure

```
medical_chatbot/
├── requirements.txt
├── README.md
├── config.py                    # Centralized hyperparameters
├── 1_prepare_data.py            # Download + format + split
├── 2_finetune.py                # Fine-tuning with Unsloth
├── 2_finetune_fallback.py       # Fine-tuning without Unsloth (Windows)
├── 3_evaluate.py                # Comparative metrics
├── 4_app.py                     # Gradio interface
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── outputs/
│   ├── lora_adapter/            # Saved LoRA adapter
│   └── results.json             # Evaluation results
└── utils/
    ├── inference.py             # Model loading and generation
    └── metrics.py               # NLP metrics computation
```

---

## OOM Troubleshooting

If you get a CUDA Out-of-Memory error during fine-tuning, try these fixes in order:

**1. Reduce `MAX_SEQ_LENGTH` in `config.py`:**
```python
MAX_SEQ_LENGTH = 1024  # instead of 2048
```

**2. Reduce batch size in `config.py`:**
```python
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8   # double to compensate
```

**3. Reduce LoRA rank:**
```python
LORA_R = 8  # instead of 16
```

**4. Clear GPU cache before launching:**
```python
import torch; torch.cuda.empty_cache()
```

**5. Gradient checkpointing** is already enabled by default in the config.

---

## Hugging Face Authentication

Some datasets may require a token:
```bash
huggingface-cli login
```

---

## Accessing the Interface

After running `python 4_app.py`, open: [http://localhost:7860](http://localhost:7860)

---

## Disclaimer

> ⚠️ This project is for **educational and research purposes only**. Model responses do not constitute professional medical advice and must not replace consultation with a qualified healthcare professional.
