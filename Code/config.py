"""训练与模型超参数配置（与论文实验设置一致）。"""

import os
from pathlib import Path

# 路径基准：本文件所在目录即 Code/ 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parent
BIYEWORK_ROOT = PROJECT_ROOT.parent.parent.parent

# 数据与训练
N_TRAJECTORIES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 2
INPUT_DIM = 4

# 模型结构
MAX_RANK = 8  # 最大内在注意模式数 R

# 损失函数
LABEL_SMOOTHING = 0.1  # ε
ORDER_LOSS_WEIGHT = 0.01  # λ_o
SUM_CONSTRAINT_WEIGHT = 0.01  # λ_s
SPARSITY_LAMBDA_INIT = 0.01  # λ(t) 前 50 epoch
SPARSITY_LAMBDA_MAX = 0.08  # λ(t) 余弦上限
SPARSITY_WARMUP_EPOCHS = 50

# 优化器
LR_WARMUP_EPOCHS = 50
LR_MIN = 1e-6
RANK_PREDICTOR_LR_MULTIPLIER = 1.5
RANK_PREDICTOR_WEIGHT_DECAY = 0.002
MAIN_WEIGHT_DECAY = 1e-3

# 早停与模型选择
EARLY_STOPPING_PATIENCE = 150

# 评估
DECISION_THRESHOLD = 0.5

# 计算复杂度 benchmark
BENCHMARK_WARMUP = 50
BENCHMARK_TIMED_RUNS = 200
BENCHMARK_BATCH_SIZE = 1

# 默认路径（相对工程目录，可通过环境变量覆盖）
_DEFAULT_OUTPUT_DIR = PROJECT_ROOT.parent / "150" / "5"
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", str(BIYEWORK_ROOT / "new_preprocessed"))
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR))
DEFAULT_FOLD_JSON = str(PROJECT_ROOT / "spilt" / "subject_fold_train_val_test.json")
DEFAULT_TENSORBOARD_DIR = os.environ.get(
    "TENSORBOARD_DIR",
    os.environ.get("OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)),
)

RANDOM_SEED = 39
