import numpy as np
import pytest

from spectronet.config import CLASSES, ModelConfig
from spectronet.models.fusion_model import BASELINE_BUILDERS
from spectronet.models.lstm_head import build_lstm_head


def test_lstm_head_output_shape():
    cfg = ModelConfig(lstm_units=[16, 8], dense_units=16, dropout=0.1)
    model = build_lstm_head(input_dim=32, model_cfg=cfg, num_classes=len(CLASSES))
    x = np.random.randn(4, 32).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (4, len(CLASSES))
    np.testing.assert_allclose(y.sum(axis=1), np.ones(4), rtol=1e-4)


@pytest.mark.parametrize("name", list(BASELINE_BUILDERS.keys()))
def test_baseline_models_build_and_predict(name):
    builder = BASELINE_BUILDERS[name]
    T, F = 20, 13
    model = builder(input_shape=(T, F), num_classes=len(CLASSES))
    x = np.random.randn(2, T, F).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, len(CLASSES))
