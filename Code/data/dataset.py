import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import N_TRAJECTORIES
from data.trajectory import augment_trajectory, load_trajectory


class ModifiedCustomDataset(Dataset):
    def __init__(self, samples, N=N_TRAJECTORIES, training=False):
        self.samples = samples
        self.N = N
        self.training = training

        self.static_features = {}
        for sample in samples:
            subject_id = sample['subject_id']
            traj_path = sample['trajectories'][0]['file']
            base_dir = None
            current_dir = os.path.dirname(traj_path)
            while current_dir != '/':
                if os.path.exists(os.path.join(current_dir, f"{subject_id}.csv")):
                    base_dir = current_dir
                    break
                current_dir = os.path.dirname(current_dir)

            if base_dir is None:
                raise ValueError(f"Cannot find csv file for subject {subject_id}")

            csv_path = os.path.join(base_dir, f"{subject_id}.csv")
            if not os.path.exists(csv_path):
                raise ValueError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            required_cols = ['File Prefix', 'NF', 'IFD', 'TFD', 'SA', 'BP']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

            df[['NF', 'IFD', 'TFD', 'SA', 'BP']] = df[['NF', 'IFD', 'TFD', 'SA', 'BP']].fillna(0)
            self.static_features[subject_id] = df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sad_trajectories = sample['trajectories'][:self.N]
        neutral_trajectories = sample['trajectories'][self.N:]

        sad_trajs = []
        sad_static_features = []
        for traj_data in sad_trajectories:
            trajectory = load_trajectory(traj_data['file'])
            if self.training:
                trajectory = augment_trajectory(trajectory)
            sad_trajs.append(trajectory)

            file_prefix = os.path.basename(traj_data['file']).split('_efix')[0]
            static_data = self.static_features[sample['subject_id']]

            matching_rows = static_data[static_data['File Prefix'] == file_prefix][
                ['NF', 'IFD', 'TFD', 'SA', 'BP']]
            if len(matching_rows) == 0:
                print(f"Warning: No matching static features found for {file_prefix}, using zeros")
                static_feat = np.zeros(5)
            else:
                static_feat = matching_rows.values[0]
                static_feat = np.nan_to_num(static_feat, 0)

            sad_static_features.append(torch.tensor(static_feat, dtype=torch.float32))

        neutral_trajs = []
        neutral_static_features = []
        for traj_data in neutral_trajectories:
            trajectory = load_trajectory(traj_data['file'])
            if self.training:
                trajectory = augment_trajectory(trajectory)
            neutral_trajs.append(trajectory)

            file_prefix = os.path.basename(traj_data['file']).split('_efix')[0]
            static_data = self.static_features[sample['subject_id']]

            matching_rows = static_data[static_data['File Prefix'] == file_prefix][
                ['NF', 'IFD', 'TFD', 'SA', 'BP']]
            if len(matching_rows) == 0:
                print(f"Warning: No matching static features found for {file_prefix}, using zeros")
                static_feat = np.zeros(5)
            else:
                static_feat = matching_rows.values[0]
                static_feat = np.nan_to_num(static_feat, 0)

            neutral_static_features.append(torch.tensor(static_feat, dtype=torch.float32))

        sad_static_features = torch.stack(sad_static_features)
        neutral_static_features = torch.stack(neutral_static_features)

        label = torch.tensor(int(sample['label']), dtype=torch.long)
        return (sad_trajs, neutral_trajs, sad_static_features, neutral_static_features), label
