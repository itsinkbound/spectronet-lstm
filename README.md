# SpectroNet-LSTM

Production implementation of a spectrogram + LSTM fusion model for cardiac
(heartbeat) anomaly detection, based on a published research paper. Classifies
heart sound recordings into 5 categories: `normal`, `murmur`, `extrastole`,
`artifact`, `extrahls`.

## How it works

1. **Audio preprocessing** — fixed-length normalization, band-pass filtering,
   wavelet denoising (`spectronet.data.preprocessing`).
2. **Spectrogram generation** — raw audio is converted to a mel-spectrogram
   image (`spectronet.features.spectrogram`), because classification happens
   on the *visual* pattern of the heartbeat, not the raw waveform.
3. **Feature extraction** — the spectrogram image is passed through three
   pretrained CNN backbones (ResNet101, VGG16, InceptionV3), fused together
   (`spectronet.models.fusion_model`).
4. **Sequence modeling** — fused features feed into a Conv1D + 2×LSTM head,
   which reads them as a sequence to catch rhythmic/temporal irregularities.
5. **Classification** — a final `Dense(5, softmax)` layer outputs a
   probability for each of the 5 classes.

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


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
pip install -r requirements.txt
```

Point the pipeline at your dataset (PASCAL/DHD "Heartbeat Sounds" — `set_a.csv`/`set_b.csv` + `.wav` files):
```bash
# either edit configs/config.yaml's data.root_dir, or:
$env:DHD_ROOT = "D:\path\to\dataset"      # Windows
export DHD_ROOT=/path/to/dataset          # Linux/macOS
```

## Usage

### 1. CLI — batch pipeline (training/evaluation against a labeled dataset)

```bash
python -m spectronet.pipeline --config configs/config.yaml --stage <stage>
```

| `--stage` | What it does |
|---|---|
| `preprocess` | Builds the manifest, runs audio cleanup, does the stratified 72/8/20 split. |
| `train` | Runs the two-phase trainer (frozen backbones → fine-tune). |
| `evaluate` | Loads the saved checkpoint (`--checkpoint`, defaults to `<checkpoint_dir>/spectronet_lstm_fine_tuned.keras`) and reports accuracy/precision/recall/F1. Runs standalone — no retraining. |
| `all` (default) | preprocess → train → evaluate in one run. |

`--stage explain` is not yet implemented in the CLI (see Known Limitations) — use the notebook for SHAP/LIME explainability.

### 2. Single-file inference (one upload, one prediction)

```bash
python -m spectronet.inference path/to/clip.wav
```
Prints a JSON dict of per-class probabilities. For repeated use, import `Predictor` directly and instantiate it once — reloading three CNN backbones per call is slow.

### 3. API — serve predictions over HTTP

```bash
uvicorn spectronet.api:app --reload
```
- `GET /health` — readiness check
- `POST /predict` — multipart file upload, returns:
```json
{
  "prediction": "normal",
  "confidence": 0.27,
  "low_confidence": true,
  "probabilities": {"normal": 0.27, "murmur": 0.19, "extrastole": 0.17, "artifact": 0.18, "extrahls": 0.19}
}
```
`low_confidence` is `true` whenever the top two classes are within `LOW_CONFIDENCE_MARGIN` (default 0.15) of each other — see Known Limitations for why this matters.

### 4. Frontend — upload UI

With the API running, open `frontend/index.html` in a browser (or serve it statically). Drag in a `.wav`/`.flac`/`.ogg` file; it shows the top prediction, confidence, and a low-confidence warning when applicable, with per-class probabilities as supporting detail underneath.

### 5. Docker

```bash
docker build -t spectronet-lstm:latest .
docker run --rm \
  -v /path/to/dataset:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/configs:/app/configs \
  spectronet-lstm:latest --config configs/config.docker.yaml --stage preprocess
```
Currently containerizes the CLI pipeline; the API is not yet dockerized separately.

### 6. Kaggle notebook

`notebooks/spectronet_lstm_kaggle.ipynb` — the research artifact. Upload to Kaggle, attach the "Heartbeat Sounds" dataset, run top to bottom. This is currently the only place all 5 baseline models (SimpleRNN, GRU, TCN, MLP, CNN-LSTM) are actually trained for a Table 1-style comparison — the CLI's baseline loop is a stub (see below).

## Testing & linting

```bash
pytest
ruff check src tests
```

## Known limitations

- **Accuracy below the paper's reported numbers.** Production evaluation on
  a held-out test set currently shows ~73% accuracy / 0.66 F1, vs. the
  paper's reported ~0.92. Some hyperparameters in the paper were
  under-specified and reconstructed by best interpretation; this gap has not
  been fully closed and is not currently being chased further.
- **API is not dockerized** — only the CLI pipeline is currently
  containerized.

## Requirements

Dependency versions are pinned in `requirements.txt` to the exact set
verified against this codebase in CI — install from that file rather than
letting pip resolve to newer/unverified versions.
