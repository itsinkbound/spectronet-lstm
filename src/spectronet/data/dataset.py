"""Loader for the Dangerous Heartbeat Dataset (DHD).

Expected layout (as published on Kaggle, "Heartbeat Sounds" / PASCAL
challenge, which the paper cites as its source):

    root_dir/
        set_a.csv
        set_b.csv
        set_a_timing.csv
        set_a/*.wav
        set_b/*.wav

`set_a.csv` / `set_b.csv` contain columns `fname,label,sublabel` (label is
one of normal/murmur/extrastole/artifact/extrahls, with some rows unlabeled
and dropped here).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from spectronet.config import CLASSES, DataConfig


@dataclasses.dataclass
class Sample:
    path: str
    label: str
    source_set: str


def load_manifest(data_cfg: DataConfig) -> pd.DataFrame:
    """Build a unified (path, label) manifest from set_a.csv + set_b.csv."""
    root = Path(data_cfg.root_dir)
    frames = []
    for set_name in ("set_a", "set_b"):
        csv_path = root / f"{set_name}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].str.strip().str.lower()
        df = df[df["label"].isin(data_cfg.classes)]

        def _resolve(fname: str) -> str:
            fname = Path(str(fname))
            name = fname.name

            # 1. Try the path exactly as provided by the CSV
            candidate = root / fname
            if candidate.exists():
                return str(candidate)

            # 2. Try just the filename inside the set directory
            candidate = root / set_name / name
            if candidate.exists():
                return str(candidate)

            # 3. Handle Set-B naming differences
            if set_name == "set_b" and name.startswith("Btraining_"):
                remainder = name[len("Btraining_"):]

                # Case A:
                # Btraining_extrastole_127_...
                # -> extrastole__127_...
                #
                # Case B:
                # Btraining_murmur_Btraining_noisymurmur_135_...
                # -> murmur_noisymurmur_135_...

                if remainder.startswith("murmur_Btraining_"):
                    new_name = "murmur_" + remainder[len("murmur_Btraining_"):]

                    candidate = root / set_name / new_name
                    if candidate.exists():
                        return str(candidate)

                elif remainder.startswith("normal_Btraining_"):
                    new_name = "normal_" + remainder[len("normal_Btraining_"):]

                    candidate = root / set_name / new_name
                    if candidate.exists():
                        return str(candidate)

                else:
                    # Generic mapping:
                    # Btraining_<label>_<rest>
                    # -> <label>__<rest>
                    parts = remainder.split("_", 1)

                    if len(parts) == 2:
                        label, rest = parts
                        new_name = f"{label}__{rest}"

                        candidate = root / set_name / new_name
                        if candidate.exists():
                            return str(candidate)

            # Return a path that will be filtered out if it doesn't exist
            return str(root / set_name / name)

        df["path"] = df["fname"].apply(_resolve)
        df["source_set"] = set_name
        frames.append(df[["path", "label", "source_set"]])

    if not frames:
        raise FileNotFoundError(
            f"No set_a.csv/set_b.csv found under {root}. "
            "Point data.root_dir at a DHD extraction."
        )
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["path"].apply(lambda p: Path(p).exists())].reset_index(
        drop=True
    )
    return manifest


def stratified_split(
    manifest: pd.DataFrame, data_cfg: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """72/8/20 stratified train/val/test split (per Section 3 of the paper)."""
    rng = np.random.RandomState(data_cfg.random_seed)
    train_frames, val_frames, test_frames = [], [], []

    for label, group in manifest.groupby("label"):
        idx = group.sample(frac=1.0, random_state=rng.randint(0, 1_000_000)).index
        n = len(idx)
        n_test = int(round(n * data_cfg.test_split))
        n_val = int(round(n * data_cfg.val_split))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        train_frames.append(group.loc[train_idx])
        val_frames.append(group.loc[val_idx])
        test_frames.append(group.loc[test_idx])

    train_df = pd.concat(train_frames).sample(frac=1.0, random_state=data_cfg.random_seed)
    val_df = pd.concat(val_frames).sample(frac=1.0, random_state=data_cfg.random_seed)
    test_df = pd.concat(test_frames).sample(frac=1.0, random_state=data_cfg.random_seed)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def label_to_index(label: str) -> int:
    return CLASSES.index(label)


def one_hot(label: str, num_classes: int = len(CLASSES)) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[label_to_index(label)] = 1.0
    return vec


def class_distribution(manifest: pd.DataFrame) -> pd.Series:
    return manifest["label"].value_counts(normalize=True).reindex(CLASSES).fillna(0.0)
