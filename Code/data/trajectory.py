import random

import torch
import torch.nn.functional as F

from config import N_TRAJECTORIES


def load_trajectory(file_path):
    """加载注视点轨迹，返回原始长度的tensor"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        traj = [[float(item.split(": ")[1]) if item.split(": ")[1] != "." else 0
                 for item in line.split(", ")[2:]]
                for line in lines]

    traj_tensor = torch.tensor(traj, dtype=torch.float32)
    return traj_tensor


def pad_trajectories(trajectory_list, n_trajectories=N_TRAJECTORIES, target_len=8):
    feature_dim = None
    for batch_item in trajectory_list:
        if isinstance(batch_item, (list, tuple)):
            for traj in batch_item:
                if torch.is_tensor(traj):
                    feature_dim = traj.size(1)
                    break
        if feature_dim is not None:
            break

    if feature_dim is None:
        raise ValueError("Could not determine feature dimension")

    padded_trajs = []
    lengths = [[] for _ in range(n_trajectories)]

    for batch_idx, batch_item in enumerate(trajectory_list):
        if not isinstance(batch_item, (list, tuple)):
            continue

        batch_trajs = []
        for traj_idx, traj in enumerate(batch_item):
            if not torch.is_tensor(traj):
                continue

            traj_len = len(traj)
            lengths[traj_idx].append(traj_len)

            if traj_len == target_len:
                processed_traj = traj
            elif traj_len < target_len:
                traj_unsqueezed = traj.unsqueeze(0)
                processed_traj = F.interpolate(
                    traj_unsqueezed.transpose(1, 2),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
            else:
                window_size = traj_len // target_len
                remainder = traj_len % target_len
                result = []

                start_idx = 0
                for i in range(target_len):
                    current_window = window_size + (1 if i < remainder else 0)
                    end_idx = start_idx + current_window
                    window_mean = traj[start_idx:end_idx].mean(dim=0)
                    result.append(window_mean)
                    start_idx = end_idx

                processed_traj = torch.stack(result)

            batch_trajs.append(processed_traj)

        if batch_trajs:
            padded_trajs.append(torch.stack(batch_trajs))

    if not padded_trajs:
        raise ValueError("No trajectories were processed")

    padded_trajs = torch.stack(padded_trajs)
    return padded_trajs


def custom_collate_fn(batch):
    trajs_and_features, labels = zip(*batch)
    sad_trajs, neutral_trajs, sad_static, neutral_static = zip(*trajs_and_features)

    sad_padded = pad_trajectories(sad_trajs)
    neutral_padded = pad_trajectories(neutral_trajs)

    sad_static = torch.stack(sad_static)
    neutral_static = torch.stack(neutral_static)
    labels = torch.stack(labels)

    return (sad_padded, neutral_padded, sad_static, neutral_static), labels


def augment_trajectory(traj, p=0.5):
    """轨迹数据增强"""
    if random.random() < p:
        aug_type = random.choice(['noise', 'scale', 'shift'])
        if aug_type == 'noise':
            noise = torch.randn_like(traj) * 0.05
            return traj + noise
        elif aug_type == 'scale':
            scale = random.uniform(0.95, 1.05)
            return traj * scale
        else:
            shift = random.uniform(-0.05, 0.05)
            return traj + shift
    return traj
