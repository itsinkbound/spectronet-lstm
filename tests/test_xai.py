import numpy as np
import pytest

from spectronet.explainability.xai import save_explanation_grid


def test_save_explanation_grid_writes_file(tmp_path):
    images = {
        "original": (np.random.rand(32, 32, 3) * 255).astype(np.uint8),
        "lime": (np.random.rand(32, 32, 3) * 255).astype(np.uint8),
    }
    path = save_explanation_grid(images, output_dir=str(tmp_path), filename="test.png")
    assert (tmp_path / "test.png").exists()
    assert path.endswith("test.png")


def test_explain_with_lime_importable():
    pytest.importorskip("lime")
    from spectronet.explainability.xai import explain_with_lime  # noqa: F401


def test_explain_with_shap_importable():
    pytest.importorskip("shap")
    from spectronet.explainability.xai import explain_with_shap  # noqa: F401
