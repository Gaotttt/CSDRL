"""Training and model hyper-parameters (aligned with the paper setup)."""

import os
from pathlib import Path

# Path anchors: this file lives in Code/; repo root is CSDRL-main/
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# Data and training
N_TRAJECTORIES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 2
INPUT_DIM = 4

# Model architecture
MAX_RANK = 8  # max number of intrinsic attentional patterns R

# Loss
LABEL_SMOOTHING = 0.1  # epsilon
ORDER_LOSS_WEIGHT = 0.01  # lambda_o
SUM_CONSTRAINT_WEIGHT = 0.01  # lambda_s
SPARSITY_LAMBDA_INIT = 0.01  # lambda(t) for the first 50 epochs
SPARSITY_LAMBDA_MAX = 0.08  # lambda(t) cosine upper bound
SPARSITY_WARMUP_EPOCHS = 50

# Optimizer
LR_WARMUP_EPOCHS = 50
LR_MIN = 1e-6
RANK_PREDICTOR_LR_MULTIPLIER = 1.5
RANK_PREDICTOR_WEIGHT_DECAY = 0.002
MAIN_WEIGHT_DECAY = 1e-3

# Early stopping and model selection
EARLY_STOPPING_PATIENCE = 150

# Evaluation
DECISION_THRESHOLD = 0.5

# Computational complexity benchmark
BENCHMARK_WARMUP = 50
BENCHMARK_TIMED_RUNS = 200
BENCHMARK_BATCH_SIZE = 1

# Default paths (relative to repo root CSDRL-main/; overridable via env vars)
_DEFAULT_OUTPUT_DIR = REPO_ROOT / "150" / "5"
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", str(REPO_ROOT / "dataset"))
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR))
DEFAULT_FOLD_JSON = str(PROJECT_ROOT / "spilt" / "subject_fold_train_val_test.json")
DEFAULT_TENSORBOARD_DIR = os.environ.get(
    "TENSORBOARD_DIR",
    os.environ.get("OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)),
)

RANDOM_SEED = 39
