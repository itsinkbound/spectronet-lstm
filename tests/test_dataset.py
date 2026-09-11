# tests/test_dataset.py
import pandas as pd
from spectronet.config import DataConfig
from spectronet.data.dataset import load_manifest

def test_load_manifest_reads_both_sets(tmp_path):
    # recreate whatever nesting/casing your real fix handles
    (tmp_path / "set_a").mkdir()
    (tmp_path / "set_b").mkdir()
    (tmp_path / "set_a" / "a1.wav").touch()
    (tmp_path / "set_b" / "b1.wav").touch()

    pd.DataFrame({"fname": ["a1.wav"], "label": ["normal"]}).to_csv(tmp_path / "set_a.csv", index=False)
    pd.DataFrame({"fname": ["b1.wav"], "label": ["murmur"]}).to_csv(tmp_path / "set_b.csv", index=False)

    manifest = load_manifest(DataConfig(root_dir=str(tmp_path)))
    assert len(manifest) == 2
    assert set(manifest["label"]) == {"normal", "murmur"}