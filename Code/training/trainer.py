import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DECISION_THRESHOLD,
    EARLY_STOPPING_PATIENCE,
    HIDDEN_DIM,
    INPUT_DIM,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LR_MIN,
    LR_WARMUP_EPOCHS,
    MAIN_WEIGHT_DECAY,
    N_TRAJECTORIES,
    NUM_EPOCHS,
    NUM_LAYERS,
    OUTPUT_DIM,
    RANK_PREDICTOR_LR_MULTIPLIER,
    RANK_PREDICTOR_WEIGHT_DECAY,
    SPARSITY_LAMBDA_INIT,
    SPARSITY_LAMBDA_MAX,
    SPARSITY_WARMUP_EPOCHS,
    ORDER_LOSS_WEIGHT,
    SUM_CONSTRAINT_WEIGHT,
    MAX_RANK,
)
from data.dataset import ModifiedCustomDataset
from data.preprocessing import create_sliding_window_samples
from data.trajectory import custom_collate_fn
from models.modified_model import ModifiedModel
from training.early_stopping import ImprovedEarlyStopping
from training.losses import LabelSmoothingLoss
from utils.metrics import (
    compute_classification_metrics,
    compute_mean_std,
    compute_subject_metrics,
    format_mean_std,
)

SAMPLE_METRIC_KEYS = ['acc', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc']
SUBJECT_METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc']
METRIC_DISPLAY_NAMES = {
    'acc': 'Accuracy',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 Score',
    'pr_auc': 'PR-AUC',
    'roc_auc': 'ROC-AUC',
}


def count_label_dist(samples):
    em = sum(1 for s in samples if s['label'] == 1)
    hp = sum(1 for s in samples if s['label'] == 0)
    return em, hp


def adjust_sparsity_lambda(epoch):
    if epoch < SPARSITY_WARMUP_EPOCHS:
        return SPARSITY_LAMBDA_INIT
    progress = min((epoch - SPARSITY_WARMUP_EPOCHS) / (NUM_EPOCHS - SPARSITY_WARMUP_EPOCHS), 1.0)
    return SPARSITY_LAMBDA_INIT + (SPARSITY_LAMBDA_MAX - SPARSITY_LAMBDA_INIT) * (
        1 - np.cos(progress * np.pi)
    ) / 2


def lr_lambda_epoch(epoch):
    """Linear warm-up + cosine annealing over each param group's base LR."""
    if epoch < LR_WARMUP_EPOCHS:
        return (epoch + 1) / LR_WARMUP_EPOCHS
    progress = min((epoch - LR_WARMUP_EPOCHS) / max(1, NUM_EPOCHS - LR_WARMUP_EPOCHS), 1.0)
    min_factor = LR_MIN / LEARNING_RATE
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + np.cos(np.pi * progress))


def build_samples(subject_ids, subject_data):
    samples = []
    for subj in subject_ids:
        samples.extend(create_sliding_window_samples(
            {subj: subject_data[subj]}, N=N_TRAJECTORIES))
    return samples


def print_sample_distribution(train_samples, val_samples, test_samples):
    train_mdd, train_hc = count_label_dist(train_samples)
    val_mdd, val_hc = count_label_dist(val_samples)
    test_mdd, test_hc = count_label_dist(test_samples)

    print("\nSample-level distribution:")
    print("Training samples:")
    print(f"MDD: {train_mdd} ({train_mdd / (train_mdd + train_hc):.2f})")
    print(f"HC: {train_hc} ({train_hc / (train_mdd + train_hc):.2f})")
    print(f"Total: {len(train_samples)}")

    print("Validation samples (early stopping / best model):")
    print(f"MDD: {val_mdd} ({val_mdd / (val_mdd + val_hc):.2f})")
    print(f"HC: {val_hc} ({val_hc / (val_mdd + val_hc):.2f})")
    print(f"Total: {len(val_samples)}")

    print("Test samples (final evaluation):")
    print(f"MDD: {test_mdd} ({test_mdd / (test_mdd + test_hc):.2f})")
    print(f"HC: {test_hc} ({test_hc / (test_mdd + test_hc):.2f})")
    print(f"Total: {len(test_samples)}")


def evaluate_loader(model, data_loader, criterion, device, samples, collect_misclassified=False):
    """Evaluate the model on the given data loader."""
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    labels_list = []
    all_preds = []
    all_trues = []
    all_sub_ids = []
    all_probs = []
    misclassified_samples = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data]
            labels = labels.to(device)

            outputs, reg_loss = model(data, training=False)
            loss = criterion(outputs, labels) + reg_loss
            total_loss += loss.item() * labels.size(0)
            total_reg_loss += reg_loss.item() * labels.size(0)

            probabilities = F.softmax(outputs[:, :2], dim=1)
            predicted = (probabilities[:, 1] >= DECISION_THRESHOLD).long()

            labels_list.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probabilities[:, 1].cpu().numpy())

            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + labels.size(0)
            batch_sub_ids = [s['subject_id'] for s in samples[start_idx:end_idx]]
            all_trues.extend(labels.cpu().numpy().tolist())
            all_sub_ids.extend(batch_sub_ids)

            if collect_misclassified:
                mistakes = predicted != labels
                if mistakes.any():
                    for i in range(len(labels)):
                        if mistakes[i]:
                            misclassified_samples.append({
                                'true_label': labels[i].item(),
                                'predicted_label': predicted[i].item(),
                                'prediction_prob': probabilities[i].cpu().numpy()
                            })

    n = len(data_loader.dataset)
    cls_metrics = compute_classification_metrics(
        np.array(labels_list),
        np.array(all_probs),
    )
    subject_metrics = compute_subject_metrics(all_sub_ids, all_trues, all_probs)

    return {
        'loss': total_loss / n,
        'reg_loss': total_reg_loss / n,
        **cls_metrics,
        'subject_metrics': subject_metrics,
        'misclassified_samples': misclassified_samples,
    }


def _metrics_to_dict(metrics, keys):
    return {key: metrics[key] for key in keys}


def train_one_fold(fold_info, subject_data, base_output_dir, writer, k_folds):
    fold_idx = fold_info['fold_idx']
    print(f"\nFold {fold_idx + 1}")
    output_dir = os.path.join(base_output_dir, f'epoch_outputs_{fold_idx + 1}')
    os.makedirs(output_dir, exist_ok=True)

    train_subjects = fold_info['train_subjects']
    val_subjects = fold_info['val_subjects']
    test_subjects = fold_info['test_subjects']

    train_samples = build_samples(train_subjects, subject_data)
    val_samples = build_samples(val_subjects, subject_data)
    test_samples = build_samples(test_subjects, subject_data)

    print_sample_distribution(train_samples, val_samples, test_samples)

    train_dataset = ModifiedCustomDataset(train_samples, N=N_TRAJECTORIES, training=True)
    val_dataset = ModifiedCustomDataset(val_samples, N=N_TRAJECTORIES, training=False)
    test_dataset = ModifiedCustomDataset(test_samples, N=N_TRAJECTORIES, training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, N=N_TRAJECTORIES)
    model = model.to(device)

    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)

    main_params = []
    rank_predictor_params = []
    for name, param in model.named_parameters():
        if 'rank_predictor' in name:
            rank_predictor_params.append(param)
        else:
            main_params.append(param)

    optimizer = optim.Adam([
        {'params': main_params, 'lr': LEARNING_RATE, 'weight_decay': MAIN_WEIGHT_DECAY},
        {
            'params': rank_predictor_params,
            'lr': LEARNING_RATE * RANK_PREDICTOR_LR_MULTIPLIER,
            'weight_decay': RANK_PREDICTOR_WEIGHT_DECAY,
        }
    ])

    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_epoch)

    early_stopping = ImprovedEarlyStopping(patience=EARLY_STOPPING_PATIENCE, window_size=5, min_delta=1e-4)

    best_val_score = None
    best_val_epoch = 0
    save_path = os.path.join(base_output_dir, f'best_model_fold_{fold_idx + 1}.pth')

    for epoch in range(NUM_EPOCHS):
        current_sparsity_lambda = adjust_sparsity_lambda(epoch)
        model.update_sparsity_lambda(current_sparsity_lambda)
        model.train()
        train_loss = 0
        train_regularization_loss = 0
        train_correct = 0

        for data, labels in train_loader:
            data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in data]
            labels = labels.to(device)

            model_device = next(model.parameters()).device
            for i, x in enumerate(data):
                if torch.is_tensor(x) and x.device != model_device:
                    data[i] = x.to(model_device)

            optimizer.zero_grad()
            outputs, total_reg_loss = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if torch.isnan(total_reg_loss).any() or torch.isinf(total_reg_loss).any():
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels) + total_reg_loss

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                optimizer.zero_grad()
                continue

            loss.backward()

            grad_has_nan = any(
                param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                for param in model.parameters()
            )
            if grad_has_nan:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_regularization_loss += total_reg_loss.item() * labels.size(0)
            train_probs = F.softmax(outputs[:, :2], dim=1)[:, 1]
            train_correct += ((train_probs >= DECISION_THRESHOLD).long() == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_regularization_loss /= len(train_loader.dataset)
        scheduler.step()

        val_result = evaluate_loader(model, val_loader, criterion, device, val_samples)
        sm = val_result['subject_metrics']

        should_stop, is_best, val_score = early_stopping.step(
            val_result['acc'], val_result['pr_auc'], val_result['f1'], epoch + 1
        )

        writer.add_scalar(f'sparsity_lambda/fold_{fold_idx + 1}', current_sparsity_lambda, epoch)
        writer.add_scalar(f'Loss/train_fold_{fold_idx + 1}', train_loss, epoch)
        writer.add_scalar(f'Loss/val_fold_{fold_idx + 1}', val_result['loss'], epoch)
        writer.add_scalar(f'train_regularization_loss/train_fold_{fold_idx + 1}', train_regularization_loss, epoch)
        writer.add_scalar(f'val_regularization_loss/val_fold_{fold_idx + 1}', val_result['reg_loss'], epoch)
        writer.add_scalar(f'Accuracy/train_fold_{fold_idx + 1}', train_acc, epoch)
        writer.add_scalar(f'Accuracy/val_fold_{fold_idx + 1}', val_result['acc'], epoch)
        writer.add_scalar(f'Precision/val_fold_{fold_idx + 1}', val_result['precision'], epoch)
        writer.add_scalar(f'Recall/val_fold_{fold_idx + 1}', val_result['recall'], epoch)
        writer.add_scalar(f'F1/val_fold_{fold_idx + 1}', val_result['f1'], epoch)
        writer.add_scalar(f'PR-AUC/val_fold_{fold_idx + 1}', val_result['pr_auc'], epoch)
        writer.add_scalar(f'ROC-AUC/val_fold_{fold_idx + 1}', val_result['roc_auc'], epoch)
        writer.add_scalar(f'ValScore/val_fold_{fold_idx + 1}', val_score, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print(
            f'Fold [{fold_idx + 1}/{k_folds}] Epoch [{epoch + 1}/{NUM_EPOCHS}], '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_result["loss"]:.4f}, '
            f'Acc(Train): {train_acc:.2f}, Acc(Val): {val_result["acc"]:.2f}, ValScore: {val_score:.4f}')
        print(
            f'Val - Precision: {val_result["precision"]:.2f}, Recall: {val_result["recall"]:.2f}, '
            f'F1: {val_result["f1"]:.2f}, PR-AUC: {val_result["pr_auc"]:.2f}, '
            f'ROC-AUC: {val_result["roc_auc"]:.2f}, '
            f'TP: {val_result["tp"]}, TN: {val_result["tn"]}, FP: {val_result["fp"]}, FN: {val_result["fn"]}')
        print(
            f'Val S_Acc: {sm["accuracy"]:.2f}, S_Precision: {sm["precision"]:.2f}, '
            f'S_Recall: {sm["recall"]:.2f}, S_F1: {sm["f1"]:.2f}, '
            f'S_PR-AUC: {sm["pr_auc"]:.2f}, S_ROC-AUC: {sm["roc_auc"]:.2f}')

        if is_best:
            best_val_score = val_score
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f'  -> New best model saved (ValScore={val_score:.4f}, epoch={best_val_epoch})')

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(
                f"Best ValScore: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}, "
                f"current Val Acc: {val_result['acc']:.4f}, PR-AUC: {val_result['pr_auc']:.4f}, "
                f"F1: {val_result['f1']:.4f}"
            )
            break

    print(
        f"\nLoading best model from epoch {best_val_epoch} "
        f"(ValScore={best_val_score:.4f}) for test evaluation..."
    )
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_result = evaluate_loader(
        model, test_loader, criterion, device, test_samples, collect_misclassified=True
    )
    test_sm = test_result['subject_metrics']
    test_sample_metrics = _metrics_to_dict(test_result, SAMPLE_METRIC_KEYS)
    test_subject_metrics = _metrics_to_dict(test_sm, SUBJECT_METRIC_KEYS)

    test_output_path = os.path.join(base_output_dir, f'test_results_fold_{fold_idx + 1}.json')
    with open(test_output_path, 'w') as f:
        json.dump({
            'fold_idx': fold_idx,
            'best_val_epoch': best_val_epoch,
            'best_val_score': best_val_score,
            'test_sample_metrics': test_sample_metrics,
            'test_subject_metrics': test_subject_metrics,
        }, f, indent=4, ensure_ascii=False)

    return {
        'fold_idx': fold_idx,
        'best_val_epoch': best_val_epoch,
        'best_val_score': best_val_score,
        'test_sample_metrics': test_sample_metrics,
        'test_subject_metrics': test_subject_metrics,
        'test_misclassified_samples': test_result['misclassified_samples'],
    }


def save_misclassified_samples(fold_result, misclassified_dir):
    fold_idx = fold_result['fold_idx']
    best_epoch = fold_result['best_val_epoch']
    samples = fold_result['test_misclassified_samples']

    misclassified_file = os.path.join(misclassified_dir, f'fold_{fold_idx + 1}_test_misclassified.txt')
    with open(misclassified_file, 'w') as f:
        f.write(
            f"Fold {fold_idx + 1} Test Set Misclassified Samples "
            f"(best model from val epoch {best_epoch}):\n"
        )
        f.write("=" * 80 + "\n\n")

        total_errors = len(samples)
        mdd_to_hc = sum(1 for s in samples if s['true_label'] == 1 and s['predicted_label'] == 0)
        hc_to_mdd = sum(1 for s in samples if s['true_label'] == 0 and s['predicted_label'] == 1)

        f.write("Summary Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total misclassified samples: {total_errors}\n")
        f.write(f"MDD predicted as HC: {mdd_to_hc}\n")
        f.write(f"HC predicted as MDD: {hc_to_mdd}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for idx, sample in enumerate(samples, 1):
            f.write(f"Misclassified Sample {idx}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"True Label: {'MDD' if sample['true_label'] == 1 else 'HC'}\n")
            f.write(f"Predicted Label: {'MDD' if sample['predicted_label'] == 1 else 'HC'}\n")
            f.write(
                f"Prediction Probabilities - HC: {sample['prediction_prob'][0]:.4f}, "
                f"MDD: {sample['prediction_prob'][1]:.4f}\n\n")

    print(f"Test misclassified samples for fold {fold_idx + 1} saved to {misclassified_file}")


def _collect_fold_metrics(fold_results, level_keys, metrics_key):
    collected = {key: [] for key in level_keys}
    for fold_result in fold_results:
        metrics = fold_result[metrics_key]
        for key in level_keys:
            collected[key].append(metrics[key])
    return collected


def _summarize_metrics(collected, keys):
    summary = {}
    for key in keys:
        mean, std = compute_mean_std(collected[key])
        summary[key] = {'mean': mean, 'std': std}
    return summary


def _print_metric_summary(title, summary, keys):
    print(f"\n{title}")
    for key in keys:
        mean = summary[key]['mean']
        std = summary[key]['std']
        print(f"{METRIC_DISPLAY_NAMES[key]}: {format_mean_std(mean, std)}")


def run_kfold_training(folds, subject_data, base_output_dir, writer):
    """Run K-fold cross-validation training."""
    k_folds = len(folds)
    fold_results = []

    misclassified_dir = os.path.join(base_output_dir, 'misclassified_samples')
    os.makedirs(misclassified_dir, exist_ok=True)

    for fold_info in folds:
        fold_result = train_one_fold(fold_info, subject_data, base_output_dir, writer, k_folds)
        fold_results.append(fold_result)
        save_misclassified_samples(fold_result, misclassified_dir)

        fold_idx = fold_result['fold_idx']
        tm = fold_result['test_sample_metrics']
        test_sm = fold_result['test_subject_metrics']

        print(
            f'Fold {fold_idx + 1} - Test Sample Metrics '
            f'(best val epoch: {fold_result["best_val_epoch"]}, '
            f'ValScore: {fold_result["best_val_score"]:.4f}):')
        print(
            f'  Acc: {tm["acc"]:.4f}, Precision: {tm["precision"]:.4f}, Recall: {tm["recall"]:.4f}, '
            f'F1: {tm["f1"]:.4f}, PR-AUC: {tm["pr_auc"]:.4f}, ROC-AUC: {tm["roc_auc"]:.4f}')

        print(f"\nFold {fold_idx + 1} - Test Subject-Level Metrics:")
        print("=" * 80)
        print(f"Accuracy: {test_sm['accuracy']:.4f}")
        print(f"Precision: {test_sm['precision']:.4f}")
        print(f"Recall: {test_sm['recall']:.4f}")
        print(f"F1 Score: {test_sm['f1']:.4f}")
        print(f"PR-AUC: {test_sm['pr_auc']:.4f}")
        print(f"ROC-AUC: {test_sm['roc_auc']:.4f}")
        print("=" * 80 + "\n")

    sample_collected = _collect_fold_metrics(fold_results, SAMPLE_METRIC_KEYS, 'test_sample_metrics')
    subject_collected = _collect_fold_metrics(fold_results, SUBJECT_METRIC_KEYS, 'test_subject_metrics')

    return {
        'fold_results': fold_results,
        'sample_summary': _summarize_metrics(sample_collected, SAMPLE_METRIC_KEYS),
        'subject_summary': _summarize_metrics(subject_collected, SUBJECT_METRIC_KEYS),
        'best_fold_test_subject_metrics': [
            {
                'fold_idx': r['fold_idx'],
                'best_val_epoch': r['best_val_epoch'],
                'best_val_score': r['best_val_score'],
                'subject_metrics': r['test_subject_metrics'],
            }
            for r in fold_results
        ],
    }


def save_final_results(results, base_output_dir, output_file_path):
    """Save final K-fold test-set results (mean ± std)."""
    sample_summary = results['sample_summary']
    subject_summary = results['subject_summary']

    _print_metric_summary(
        "Overall Test Sample-Level Metrics (mean ± std across 5 folds):",
        sample_summary,
        SAMPLE_METRIC_KEYS,
    )
    _print_metric_summary(
        "Overall Test Subject-Level Metrics (mean ± std across 5 folds):",
        subject_summary,
        SUBJECT_METRIC_KEYS,
    )

    summary_json_path = os.path.join(base_output_dir, 'overall_test_metrics_summary.json')
    with open(summary_json_path, 'w') as f_json:
        json.dump({
            'sample_level': sample_summary,
            'subject_level': subject_summary,
        }, f_json, indent=4, ensure_ascii=False)

    subject_metrics_path = os.path.join(base_output_dir, 'all_fold_test_subject_metrics.json')
    with open(subject_metrics_path, 'w') as f_json:
        json.dump(results['best_fold_test_subject_metrics'], f_json, indent=4, ensure_ascii=False)

    print(f"Saved overall test metrics summary to {summary_json_path}")
    print(f"Saved all folds' test subject-level metrics to {subject_metrics_path}")

    with open(output_file_path, 'w') as f:
        f.write("Model Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(f"N_TRAJECTORIES: {N_TRAJECTORIES}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
        f.write(f"NUM_LAYERS: {NUM_LAYERS}\n")
        f.write(f"OUTPUT_DIM: {OUTPUT_DIM}\n")
        f.write(f"INPUT_DIM: {INPUT_DIM}\n")
        f.write(f"MAX_RANK (R): {MAX_RANK}\n")
        f.write(f"LABEL_SMOOTHING (epsilon): {LABEL_SMOOTHING}\n")
        f.write(f"SPARSITY_LAMBDA: {SPARSITY_LAMBDA_INIT} (epoch 0-{SPARSITY_WARMUP_EPOCHS - 1}), "
                f"cosine to {SPARSITY_LAMBDA_MAX}\n")
        f.write(f"ORDER_LOSS_WEIGHT (lambda_o): {ORDER_LOSS_WEIGHT}\n")
        f.write(f"SUM_CONSTRAINT_WEIGHT (lambda_s): {SUM_CONSTRAINT_WEIGHT}\n")
        f.write(f"DECISION_THRESHOLD: {DECISION_THRESHOLD}\n")
        f.write(f"LR schedule: linear warmup {LR_WARMUP_EPOCHS} epochs + cosine annealing to {LR_MIN}\n")
        f.write(f"Rank predictor LR: {LEARNING_RATE * RANK_PREDICTOR_LR_MULTIPLIER}, "
                f"weight_decay: {RANK_PREDICTOR_WEIGHT_DECAY}\n")
        f.write(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs\n")
        f.write("Model selection: validation subset (from training fold), ValScore composite\n")

        f.write("\n" + "=" * 50 + "\n\n")

        f.write("Overall Test Subject-Level Metrics (mean ± std across 5 folds):\n")
        f.write("-" * 50 + "\n")
        for key in SUBJECT_METRIC_KEYS:
            mean = subject_summary[key]['mean']
            std = subject_summary[key]['std']
            f.write(f"{METRIC_DISPLAY_NAMES[key]}: {format_mean_std(mean, std)}\n")

        f.write("\n" + "=" * 50 + "\n\n")

        f.write("Overall Test Sample-Level Metrics (mean ± std across 5 folds):\n")
        f.write("-" * 50 + "\n")
        for key in SAMPLE_METRIC_KEYS:
            mean = sample_summary[key]['mean']
            std = sample_summary[key]['std']
            f.write(f"{METRIC_DISPLAY_NAMES[key]}: {format_mean_std(mean, std)}\n")

    print(f"Results saved to {output_file_path}")

    return sample_summary, subject_summary
