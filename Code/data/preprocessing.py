import json
import os

import pandas as pd

from config import N_TRAJECTORIES


def group_trajectories_by_subject(files, labels):
    """Group trajectories by subject."""
    subject_data = {}
    removed_mdd = 0
    removed_hc = 0

    for file_path, label in zip(files, labels):
        with open(file_path, 'r') as f:
            traj_length = len(f.readlines())
            if traj_length <= 2:
                # Positive: MDD; negative: HC
                if 'MDD' in file_path:
                    removed_mdd += 1
                else:
                    removed_hc += 1
                continue

        parts = file_path.split(os.sep)
        subject_folder = [p for p in parts if 'Baseline' in p][0]
        emotion_type = 'sad' if 'sad' in file_path else 'neutral'

        if subject_folder not in subject_data:
            subject_data[subject_folder] = {
                'sad': [],
                'neutral': [],
                'label': label
            }

        trajectory_data = {'file': file_path}
        subject_data[subject_folder][emotion_type].append(trajectory_data)

    print(f"\nRemoved trajectories with length <= 2:")
    print(f"MDD: {removed_mdd} trajectories removed")
    print(f"HC: {removed_hc} trajectories removed")

    return subject_data


def create_sliding_window_samples(subject_data, N, target_len=30):
    samples = []

    for subject_id, data in subject_data.items():
        sad_trajectories = data['sad']
        neutral_trajectories = data['neutral']

        n_sad = len(sad_trajectories)
        n_neutral = len(neutral_trajectories)

        print(f"\nSubject {subject_id}:")
        print(f"Available sad trajectories: {n_sad}")
        print(f"Available neutral trajectories: {n_neutral}")

        if n_sad == 0 or n_neutral == 0:
            print("Skipping subject: one class has 0 trajectories.")
            continue

        if n_sad < target_len:
            sad_trajectories = [
                sad_trajectories[i % n_sad] for i in range(target_len)
            ]
            print(f"Sad padded to {target_len}")
        else:
            sad_trajectories = sad_trajectories[:target_len]

        if n_neutral < target_len:
            neutral_trajectories = [
                neutral_trajectories[i % n_neutral] for i in range(target_len)
            ]
            print(f"Neutral padded to {target_len}")
        else:
            neutral_trajectories = neutral_trajectories[:target_len]

        n_pairs = target_len - N + 1
        print(f"Sliding windows: {n_pairs}")

        for i in range(n_pairs):
            sad_slice = sad_trajectories[i:i + N]
            neutral_slice = neutral_trajectories[i:i + N]

            sample = {
                'trajectories': sad_slice + neutral_slice,
                'label': data['label'],
                'subject_id': subject_id
            }
            samples.append(sample)

    print(f"\nTotal samples created: {len(samples)}")
    return samples


def verify_data_files(subject_data):
    """Verify trajectory files and static-feature CSVs exist and are well-formed."""
    print("\nVerifying data files...")
    for subject_id, data in subject_data.items():
        sample_traj = data['sad'][0]['file']
        base_dir = None
        current_dir = os.path.dirname(sample_traj)
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir, f"{subject_id}.csv")):
                base_dir = current_dir
                break
            current_dir = os.path.dirname(current_dir)

        if base_dir is None:
            raise ValueError(f"Cannot find csv file for subject {subject_id}")

        csv_path = os.path.join(base_dir, f"{subject_id}.csv")

        if not os.path.exists(csv_path):
            raise ValueError(f"Missing CSV file for subject {subject_id}: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            required_cols = ['File Prefix', 'NF', 'IFD', 'TFD', 'SA', 'BP']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {str(e)}")

    print("All data files verified successfully!")


def load_all_files(data_root):
    """Load all trajectory file paths and labels.

    Expected layout: dataset/{MDD,HC}/...
    Label convention: positive(1)=MDD, negative(0)=HC.
    """
    all_files = []
    labels = []

    class_folders = [
        ('MDD', 1.0),
        ('HC', 0.0),
    ]
    found_any = False

    for folder_name, label in class_folders:
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        found_any = True

        for dirpath, dirnames, filenames in os.walk(folder_path):
            if "Fixation point" in dirpath:
                for f in filenames:
                    if f != '.DS_Store':
                        file_path_txt = os.path.join(dirpath, f)
                        all_files.append(file_path_txt)
                        labels.append(label)

    if not found_any:
        raise FileNotFoundError(
            f"MDD/HC subdirectories not found under {data_root}"
        )

    return all_files, labels


def load_folds(json_path):
    """Load fold JSON and return all fold info (train/val/test)."""
    with open(json_path, 'r') as f:
        folds = json.load(f)
    print(f"Loaded {len(folds)} folds from {json_path}")
    for fold in folds:
        assert 'test_subjects' in fold, (
            f"Fold {fold['fold_idx']} is missing test_subjects"
        )
    return folds


def load_train_val_folds(json_path):
    """Backward-compatible alias for load_folds."""
    return load_folds(json_path)
