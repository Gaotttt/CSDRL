import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from config import DECISION_THRESHOLD


def _to_numpy(labels, predictions):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    return labels, predictions


def calculate_pr_auc(labels, predictions):
    """计算 PR-AUC (Average Precision)"""
    labels, predictions = _to_numpy(labels, predictions)
    try:
        if len(set(labels)) < 2:
            return 0.0
        return float(average_precision_score(labels, predictions))
    except ValueError:
        return 0.0


def calculate_roc_auc(labels, predictions):
    """计算 ROC-AUC"""
    labels, predictions = _to_numpy(labels, predictions)
    try:
        if len(set(labels)) < 2:
            return 0.0
        return float(roc_auc_score(labels, predictions))
    except ValueError:
        return 0.0


def calculate_auc(labels, predictions):
    """兼容旧接口，返回 PR-AUC"""
    return calculate_pr_auc(labels, predictions)


def compute_classification_metrics(labels, probs, threshold=DECISION_THRESHOLD):
    """计算样本级分类指标（固定阈值 threshold，默认 0.5）"""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    predictions = (probs >= threshold).astype(int)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    n = len(labels)
    acc = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    pr_auc = calculate_pr_auc(labels, probs)
    roc_auc = calculate_roc_auc(labels, probs)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def compute_subject_metrics(all_sub_ids, all_trues, all_probs):
    """计算 subject 级别的聚合指标"""
    subject_data_dict = {}
    for subj_id, true_label, prob in zip(all_sub_ids, all_trues, all_probs):
        if subj_id not in subject_data_dict:
            subject_data_dict[subj_id] = {
                'probs': [],
                'true_label': true_label
            }
        subject_data_dict[subj_id]['probs'].append(prob)

    subject_predictions = []
    subject_true_labels = []
    subject_agg_probs = []
    subject_avg_metrics = {}

    for subj_id, data in subject_data_dict.items():
        agg_prob = float(np.mean(data['probs']))
        pred_label = 1 if agg_prob >= DECISION_THRESHOLD else 0
        true_label = data['true_label']

        subject_predictions.append(pred_label)
        subject_true_labels.append(true_label)
        subject_agg_probs.append(agg_prob)

        subject_avg_metrics[subj_id] = {
            'accuracy': 1.0 if pred_label == true_label else 0.0,
            'agg_prob': agg_prob,
            'true_label': true_label,
            'pred_label': pred_label,
            'sample_count': len(data['probs'])
        }

    if len(subject_true_labels) > 0:
        metrics = compute_classification_metrics(
            np.array(subject_true_labels),
            np.array(subject_agg_probs),
        )
        return {
            'accuracy': metrics['acc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'pr_auc': metrics['pr_auc'],
            'roc_auc': metrics['roc_auc'],
            'detailed_metrics': subject_avg_metrics,
        }

    return {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'pr_auc': 0.0,
        'roc_auc': 0.0,
        'detailed_metrics': subject_avg_metrics,
    }


def compute_mean_std(values):
    """计算五折均值的样本标准差 (ddof=1)"""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def format_mean_std(mean, std):
    return f"{mean:.4f} ± {std:.4f}"
