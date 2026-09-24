"""Main entry: load data and run K-fold cross-validation training."""

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
    parser = argparse.ArgumentParser(
        description="Eye-tracking trajectory classification training"
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Preprocessed data root")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--fold-json", default=DEFAULT_FOLD_JSON, help="K-fold split JSON path")
    parser.add_argument(
        "--tensorboard-dir", default=DEFAULT_TENSORBOARD_DIR, help="TensorBoard log directory"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
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

    if os.environ.get("SKIP_MODEL_STATS", "0") != "1":
        compute_model_stats(args.output_dir)
    else:
        print("Skipping computational complexity benchmark (SKIP_MODEL_STATS=1)")

    all_subjects = list(subject_data.keys())
    total_mdd = sum(1 for s in all_subjects if subject_data[s]['label'] == 1)
    total_hc = sum(1 for s in all_subjects if subject_data[s]['label'] == 0)
    print("\nOriginal subject distribution:")
    print(f"Total subjects - MDD: {total_mdd}, HC: {total_hc}")

    folds = load_train_val_folds(args.fold_json)
    output_file_path = os.path.join(args.output_dir, 'overall_results.txt')

    results = run_kfold_training(folds, subject_data, args.output_dir, writer)
    writer.close()

    print("K-fold cross-validation finished")
    save_final_results(results, args.output_dir, output_file_path)


if __name__ == "__main__":
    main()
