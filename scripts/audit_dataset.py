import os
from pathlib import Path
import pandas as pd

from spectronet.config import DataConfig
from spectronet.data.dataset import load_manifest


root = Path(os.environ["DHD_ROOT"])

print("ROOT:", root)
print()


# --------------------------------------------------
# 1. Check CSV + WAV counts
# --------------------------------------------------

for set_name in ["set_a", "set_b"]:
    csv_path = root / f"{set_name}.csv"
    wav_dir = root / set_name

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    print("=" * 50)
    print(set_name.upper())
    print("=" * 50)

    print("CSV rows:", len(df))
    print("Labelled:", df["label"].notna().sum())
    print("WAV files:", len(list(wav_dir.glob("*.wav"))))

    print("\nLabels:")
    print(df["label"].value_counts(dropna=False))

    # Check whether labelled files actually exist
    labelled = df.dropna(subset=["label"])

    found = 0
    missing = 0

    for fname in labelled["fname"]:
        fname = str(fname)

        candidate1 = root / fname
        candidate2 = root / set_name / Path(fname).name

        if candidate1.exists() or candidate2.exists():
            found += 1
        else:
            missing += 1

    print("\nLabelled files found:", found)
    print("Labelled files missing:", missing)
    print()


# --------------------------------------------------
# 2. Test YOUR actual load_manifest()
# --------------------------------------------------

print("=" * 50)
print("LOAD_MANIFEST RESULT")
print("=" * 50)

cfg = DataConfig(root_dir=os.environ["DHD_ROOT"])

print("Configured classes:")
print(cfg.classes)
print()

manifest = load_manifest(cfg)

print("TOTAL MANIFEST:", len(manifest))
print()

print("By source:")
print(manifest["source_set"].value_counts())
print()

print("By source + label:")
print(manifest.groupby(["source_set", "label"]).size())