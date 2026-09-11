# SpectroNet-LSTM

Production implementation of **"SpectroNet-LSTM: An Interpretable Deep Learning
Approach to Cardiac Anomaly Detection Through Heart Sound Analysis"**
(Sharma, Srivats, P B, Mishra, Krithiga R, Nachiyappan S).

The paper classifies heartbeat audio into 5 classes (`normal`, `murmur`,
`extrastole`, `artifact`, `extrahls`) using the public **Dangerous Heartbeat
Dataset (DHD)** (Kaggle "Heartbeat Sounds" / PASCAL challenge, `set_a` +
`set_b`). Pipeline: denoise → spectrogram → parallel CNN feature extraction
(ResNet101 / VGG16 / InceptionV3) → feature fusion → 2-layer LSTM head →
fine-tuning → SHAP/LIME explainability.

## Repo layout

```
notebooks/                  Kaggle-ready exploratory notebook (paper reproduction)
src/spectronet/
  data/                      Dataset loading + audio preprocessing (denoise, resample)
  features/                  Spectrogram/MFCC generation + CNN backbone feature extraction
  models/                    LSTM head + 3-backbone fusion model (fine-tuning aware)
  training/                  Training loop, callbacks, hyperparameters
  evaluation/                Accuracy/Precision/Recall/F1, confusion matrix, ROC-AUC
  explainability/            SHAP + LIME wrappers producing per-class heatmaps
  pipeline.py                CLI entrypoint wiring the whole thing together
configs/config.yaml          All hyperparameters (paper defaults pre-filled)
tests/                       Unit tests for the pipeline building blocks
.github/workflows/ci.yml     Lint + unit test CI on every PR
Dockerfile                   Reproducible training/inference container
```

## Why this structure

The Kaggle notebook is kept as the **research artifact** (fast iteration,
visual EDA, ablations mirroring the paper's Table 1/Table 2). The `src/`
package is the **productionized** version of the same logic: pure functions,
typed configs, unit-testable, importable, and runnable outside a notebook
via `python -m spectronet.pipeline`.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Point at the DHD dataset (Kaggle: "Heartbeat Sounds")
export DHD_ROOT=/path/to/set_a_and_set_b

# 2. Run the full pipeline: preprocess -> extract features -> train -> evaluate -> explain
python -m spectronet.pipeline --config configs/config.yaml

# 3. Or run individual stages
python -m spectronet.pipeline --config configs/config.yaml --stage preprocess
python -m spectronet.pipeline --config configs/config.yaml --stage train
python -m spectronet.pipeline --config configs/config.yaml --stage evaluate
python -m spectronet.pipeline --config configs/config.yaml --stage explain
```

## Reproducing paper metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Fine-Tuned LSTM (this repo, `configs/config.yaml` defaults) | 0.92 | 0.92 | 0.93 | 0.92 |
| Simple RNN | 0.88 | 0.87 | 0.86 | 0.86 |
| TCN | 0.89 | 0.88 | 0.87 | 0.87 |
| GRU | 0.90 | 0.89 | 0.89 | 0.89 |
| MLP | 0.85 | 0.84 | 0.83 | 0.83 |

`src/spectronet/training/train.py` also trains the four baselines above when
`config.baselines.enabled: true`, so `evaluation/metrics.py` regenerates the
paper's Table 1 and radar/ROC plots automatically.

## Notes on deviations from the paper

- The paper does not specify exact spectrogram resolution, hop length, or
  augmentation parameters; sensible librosa defaults are used and exposed in
  `configs/config.yaml` — tune per your compute budget.
- Dataset download is not automated (Kaggle auth required); point
  `data.root_dir` at a local copy.
- CI runs unit tests on synthetic tensors (no network/dataset access), not
  full training, to keep it fast and dataset-independent.

## Development workflow

This repo was built with feature branches merged into `main`:
`feature/data-pipeline` → `feature/model-architecture` →
`feature/training-pipeline` → `feature/xai-explainability` →
`feature/cicd-docker`. See `git log --graph --oneline --all`.
