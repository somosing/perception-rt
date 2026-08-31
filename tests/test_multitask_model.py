import pytest
import torch
from transformers import SegformerConfig, SegformerModel

from perception_rt.models.multitask import PerceptionRTModel


def make_tiny_model() -> PerceptionRTModel:
    config = SegformerConfig(
        num_channels=3,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 1, 2, 4],
        mlp_ratios=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    encoder = SegformerModel(config)

    return PerceptionRTModel(
        encoder,
        number_of_classes=15,
        decoder_channels=16,
        dropout_probability=0.0,
    )


def test_multitask_model_produces_full_resolution_outputs() -> None:
    model = make_tiny_model()
    image = torch.randn(2, 3, 64, 128)

    output = model(image)

    assert output["semantic_logits"].shape == (2, 15, 64, 128)
    assert output["log_depth"].shape == (2, 1, 64, 128)
    assert output["depth_log_scale"].shape == (2, 1, 64, 128)

    for prediction in output.values():
        assert prediction.dtype == torch.float32
        assert torch.isfinite(prediction).all()


def test_all_tasks_and_encoder_receive_gradients() -> None:
    model = make_tiny_model()
    output = model(torch.randn(2, 3, 64, 128))

    loss = sum(prediction.square().mean() for prediction in output.values())
    loss.backward()

    modules = (
        model.encoder,
        model.semantic_decoder,
        model.depth_decoder,
        model.uncertainty_decoder,
    )

    for module in modules:
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        assert trainable
        assert any(parameter.grad is not None for parameter in trainable)


def test_multitask_model_rejects_invalid_image_shape() -> None:
    model = make_tiny_model()

    with pytest.raises(ValueError, match=r"\[B, 3, H, W\]"):
        model(torch.randn(2, 1, 64, 128))
