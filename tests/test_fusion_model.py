# tests/test_fusion_model.py
def test_fusion_model_builds_without_symbolic_tensor_error():
    from spectronet.config import ModelConfig
    from spectronet.models.fusion_model import build_fusion_model
    model = build_fusion_model(ModelConfig())
    assert model.output_shape == (None, 5)