import json
import os
import time

import numpy as np
import torch

from config import (
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_TIMED_RUNS,
    BENCHMARK_WARMUP,
    HIDDEN_DIM,
    INPUT_DIM,
    MAX_RANK,
    N_TRAJECTORIES,
    NUM_LAYERS,
    OUTPUT_DIM,
)
from models.modified_model import ModifiedModel

try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None


def _make_dummy_input(batch_size=BENCHMARK_BATCH_SIZE):
    return (
        torch.randn(batch_size, N_TRAJECTORIES, 8, INPUT_DIM),
        torch.randn(batch_size, N_TRAJECTORIES, 8, INPUT_DIM),
        torch.randn(batch_size, N_TRAJECTORIES, 5),
        torch.randn(batch_size, N_TRAJECTORIES, 5),
    )


def _benchmark_latency(model, device, warmup=BENCHMARK_WARMUP, runs=BENCHMARK_TIMED_RUNS):
    model.eval()
    dummy = _make_dummy_input(BENCHMARK_BATCH_SIZE)
    dummy = tuple(x.to(device) for x in dummy)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy, training=False)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy, training=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

    mean_ms = float(np.mean(elapsed) * 1000)
    std_ms = float(np.std(elapsed, ddof=1) * 1000) if len(elapsed) > 1 else 0.0
    return mean_ms, std_ms


def compute_model_stats(output_dir=None):
    """计算 Params、FLOPs 及 CPU/GPU 单样本推理延迟。"""
    temp_model = ModifiedModel(
        INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM,
        N=N_TRAJECTORIES, max_rank=MAX_RANK,
    )
    total_params = sum(p.numel() for p in temp_model.parameters())
    params_m = total_params / 1e6

    flops_value = None
    flops_info = "FLOPs: Not Available"
    if profile is not None:
        try:
            dummy_input = _make_dummy_input(BENCHMARK_BATCH_SIZE)
            flops, _ = profile(temp_model, inputs=(dummy_input,), verbose=False)
            flops_str, _ = clever_format([flops, total_params], "%.3f")
            flops_info = f"FLOPs: {flops_str}"
            flops_value = flops_str
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
    else:
        print("thop library not installed, skipping FLOPs calculation.")

    print(f"Total Parameters: {params_m:.4f} M ({total_params:,})")
    print(flops_info)

    latency_results = {
        'protocol': {
            'batch_size': BENCHMARK_BATCH_SIZE,
            'warmup_runs': BENCHMARK_WARMUP,
            'timed_runs': BENCHMARK_TIMED_RUNS,
        },
        'cpu': None,
        'gpu': None,
    }

    cpu_model = ModifiedModel(
        INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM,
        N=N_TRAJECTORIES, max_rank=MAX_RANK,
    ).to('cpu')
    cpu_mean, cpu_std = _benchmark_latency(cpu_model, torch.device('cpu'))
    latency_results['cpu'] = {
        'device': 'CPU (Intel Xeon, Linux)',
        'mean_ms_per_sample': cpu_mean,
        'std_ms_per_sample': cpu_std,
    }
    print(f"CPU latency: {cpu_mean:.4f} ± {cpu_std:.4f} ms/sample")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_model = ModifiedModel(
            INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM,
            N=N_TRAJECTORIES, max_rank=MAX_RANK,
        ).to('cuda')
        gpu_mean, gpu_std = _benchmark_latency(gpu_model, torch.device('cuda'))
        latency_results['gpu'] = {
            'device': gpu_name,
            'mean_ms_per_sample': gpu_mean,
            'std_ms_per_sample': gpu_std,
        }
        print(f"GPU latency ({gpu_name}): {gpu_mean:.4f} ± {gpu_std:.4f} ms/sample")
    else:
        print("GPU not available, skipping GPU latency benchmark.")

    stats = {
        'params': total_params,
        'params_M': params_m,
        'flops': flops_value,
        'latency': latency_results,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, 'computational_complexity.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"Computational complexity saved to {stats_path}")

    return stats
