"""
config.py — Centralized configuration for the Medical Chatbot project.
All hyperparameters, paths, and constants are defined here.
"""

import os

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

# ============================================================
# Model
# ============================================================
BASE_MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, "outputs")
ADAPTER_DIR    = os.path.join(OUTPUTS_DIR, "lora_adapter")
RESULTS_FILE   = os.path.join(OUTPUTS_DIR, "results.json")

TRAIN_FILE     = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE       = os.path.join(DATA_DIR, "val.jsonl")
TEST_FILE      = os.path.join(DATA_DIR, "test.jsonl")

# ============================================================
# Dataset
# ============================================================
# Primary source and ordered fallbacks
DATASET_SOURCES = [
    "lavita/MedQuAD",
    "keivalya/MedQuad-MedicalQnADataset",
    "medalpaca/medical_meadow_medqa",
]
MAX_EXAMPLES = 15_000   # cap to stay within 6 GB VRAM budget

# Train / val / test proportions
TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05

# Filtering: drop examples outside this token range
MIN_OUTPUT_LENGTH = 20   # characters
MAX_OUTPUT_LENGTH = 1500 # characters

# System prompt injected into every conversation
SYSTEM_PROMPT = (
    "You are a knowledgeable and compassionate medical assistant. "
    "You provide clear, accurate, and helpful information about medical topics. "
    "Always remind users that your answers are informational only and do not "
    "replace professional medical advice, diagnosis, or treatment. "
    "Encourage users to consult a qualified healthcare professional for personal concerns."
)

# ============================================================
# LoRA hyperparameters
# ============================================================
LORA_R           = 16
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.0          # 0 is recommended by Unsloth for speed
TARGET_MODULES   = "all-linear" # Unsloth syntax; for fallback see 2_finetune_fallback.py
LORA_BIAS        = "none"

# ============================================================
# Training hyperparameters
# ============================================================
MAX_SEQ_LENGTH       = 1024   # Reduced from 2048 to fit 6 GB VRAM
BATCH_SIZE           = 1      # Reduced from 2 to avoid OOM on long sequences
GRAD_ACCUM_STEPS     = 8      # Doubled to keep effective batch = 8
LEARNING_RATE        = 2e-4
NUM_EPOCHS           = 2
WARMUP_STEPS         = 10
LR_SCHEDULER         = "linear"
OPTIMIZER            = "adamw_8bit"
MAX_GRAD_NORM        = 1.0
WEIGHT_DECAY         = 0.01
FP16                 = False   # Use BF16 if supported
BF16                 = True    # Ampere+ GPUs support BF16

# Save a checkpoint every N steps (0 = only at end)
SAVE_STEPS           = 0
LOGGING_STEPS        = 10

# ============================================================
# Evaluation
# ============================================================
MAX_EVAL_EXAMPLES    = 50      # Reduced from 100 to avoid sleep-induced VRAM fragmentation

# ============================================================
# Generation (inference)
# ============================================================
TEMPERATURE          = 0.7
TOP_P                = 0.9
MAX_NEW_TOKENS       = 512
REPETITION_PENALTY   = 1.1
DO_SAMPLE            = True
