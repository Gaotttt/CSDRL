# CSDRL
Cross-Stimulus Disentangled Representation Learning for Depression Screening with a High-Quality Public Benchmark

Official training and evaluation code for **CSDRL**: participant-independent 5-fold CV on eye-tracking trajectories (HC vs MDD), with sample-level and subject-level metrics, early stopping on a validation split, and final reporting on a held-out test split.

## Repository layout

```
CSDRL-main/
├── Code/
│   ├── main.py                 # entry: python main.py
│   ├── config.py               # hyperparameters + default paths (CLI via argparse)
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset (trajectories + static features)
│   │   ├── preprocessing.py    # load files, subject grouping, sliding windows, folds
│   │   └── trajectory.py       # load / pad / augment trajectories, collate_fn
│   ├── models/
│   │   └── modified_model.py   # CSDRL model 
│   ├── training/
│   │   ├── trainer.py          # 5-fold CV loop, val selection, test evaluation
│   │   ├── losses.py           # label-smoothing CE
│   │   └── early_stopping.py   # sliding-window ValScore early stopping
│   ├── utils/
│   │   ├── metrics.py          # Acc / P / R / F1 / PR-AUC / ROC-AUC (+ subject-level)
│   │   └── model_stats.py      # Params, FLOPs, CPU/GPU latency benchmark
│   └── spilt/                  # fold JSON (train / val / test subject IDs)
├── dataset/                    # bundled eye-tracking data (HC / MDD)
├── LICENSE
└── README.md
```

## Quick start

```bash
cd Code
python main.py \
  --data-root ../dataset \
  --fold-json spilt/subject_fold_train_val_test.json \
  --output-dir ../outputs
```
