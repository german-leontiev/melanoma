import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from melanoma_research.model import build_model  # noqa: E402


def test_model_exposes_attention_parameters_and_binary_logit():
    model = build_model(pretrained=False).eval()
    state_keys = set(model.state_dict())

    assert "model.layer3.0.cbam.channel_att.1.weight" in state_keys
    assert "model.layer4.2.cbam.spatial_att.0.weight" in state_keys
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert tuple(output.shape) == (1, 1)
