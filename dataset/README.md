### Expected layout per subject

```text
dataset/
├── MDD/
│   └── MDD_Baseline_XXX/
│       ├── MDD_Baseline_XXX.csv         # per-trial static features (NF, IFD, TFD, SA, BP, …)
│       └── Fixation point/              # fixation trajectories used by the model
│           ├── *_sad_*_efix.txt
│           └── *_neutral_*_efix.txt
└── HC/
    └── HC_Baseline_XXX/
        ├── HC_Baseline_XXX.csv
        └── Fixation point/
            └── ...
```

### Fixation trajectory files (`*_efix.txt`)

Each text file under `Fixation point/` is one fixation sequence recorded while the subject viewed a single **stimulus image**.

**File naming follows the stimulus material name**, not the subject ID. Typical pattern:

```text
{stimulus_id}_{emotion}_{side}_efix.txt
```

Examples:

- `O07F-65_sad_left_efix.txt` — stimulus `O07F-65`, sad condition, left presentation
- `Y12F-18_neutral_right_efix.txt` — stimulus `Y12F-18`, neutral condition, right presentation

The `{emotion}` token (`sad` / `neutral`) is used to split trajectories into the two emotion streams. File contents are fixation events (duration, X/Y mean, pupil size, …).
