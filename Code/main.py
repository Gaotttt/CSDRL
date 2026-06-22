"""主程序入口：加载数据并执行 K 折交叉验证训练。"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FOLD_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TENSORBOARD_DIR,
    RANDOM_SEED,
)
from data.preprocessing import (
    group_trajectories_by_subject,
    load_all_files,
    load_train_val_folds,
    verify_data_files,
)
from training.trainer import run_kfold_training, save_final_results
from utils.model_stats import compute_model_stats


def parse_args():
    parser = argparse.ArgumentParser(description="眼动轨迹 EM/HP 分类模型训练")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="预处理数据根目录")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--fold-json", default=DEFAULT_FOLD_JSON, help="K折划分 JSON 文件路径")
    parser.add_argument("--tensorboard-dir", default=DEFAULT_TENSORBOARD_DIR, help="TensorBoard 日志目录")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_dir)

    all_files, labels = load_all_files(args.data_root)
    subject_data = group_trajectories_by_subject(all_files, labels)
    verify_data_files(subject_data)

    compute_model_stats(args.output_dir)

    all_subjects = list(subject_data.keys())
    total_em = sum(1 for s in all_subjects if subject_data[s]['label'] == 1)
    total_hp = sum(1 for s in all_subjects if subject_data[s]['label'] == 0)
    print("\nOriginal subject distribution:")
    print(f"Total subjects - EM: {total_em}, HP: {total_hp}")

    folds = load_train_val_folds(args.fold_json)
    output_file_path = os.path.join(args.output_dir, 'overall_results.txt')

    results = run_kfold_training(folds, subject_data, args.output_dir, writer)
    writer.close()

    print("K折交叉验证完成")
    save_final_results(results, args.output_dir, output_file_path)


if __name__ == "__main__":
    main()
