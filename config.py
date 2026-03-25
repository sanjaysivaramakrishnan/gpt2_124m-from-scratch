"""
Centralized training configuration for GPT-2.

All hyperparameters and paths in one place for easy tuning.
"""


# ── Data ─────────────────────────────────────────────────────────────────
DATA_DIR = "edu_fineweb10B"          # directory containing tokenized .npy shards

# ── Model (GPT-2 124M defaults) ─────────────────────────────────────────
BLOCK_SIZE = 1024
VOCAB_SIZE = 50304                   # padded to nearest multiple of 128
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768

# ── Training ─────────────────────────────────────────────────────────────
TOTAL_BATCH_SIZE = 524288            # 2**19 ≈ 0.5M tokens per step
MICRO_BATCH_SIZE = 64                # per-GPU micro-batch
SEQ_LEN = 1024                       # sequence length (= block_size)

MAX_LR = 6e-4
MIN_LR = 6e-5                       # MAX_LR * 0.1
WARMUP_STEPS = 500
MAX_STEPS = 19073                    # ~1 epoch over 10B tokens
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# ── Checkpointing ───────────────────────────────────────────────────────
LOG_DIR = "log"
EVAL_INTERVAL = 1000                 # validate & save every N steps
VAL_LOSS_STEPS = 20                  # micro-batches to average for val loss

# ── Misc ─────────────────────────────────────────────────────────────────
SEED = 2132006
USE_COMPILE = True                   # torch.compile for kernel fusion
