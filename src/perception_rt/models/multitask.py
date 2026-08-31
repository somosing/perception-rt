"""SegFormer-based multitask perception network."""

from collections.abc import Sequence
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from transformers import SegformerModel


class PerceptionOutput(TypedDict):
    """Dense predictions produced by PerceptionRT."""

    semantic_logits: Tensor
    log_depth: Tensor
    depth_log_scale: Tensor


class MultiScaleDecoder(nn.Module):
    """Fuse all MiT encoder stages for one dense prediction task."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        output_channels: int,
        *,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        decoder_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for input_channels in encoder_channels
            ]
        )

        fused_channels = len(encoder_channels) * decoder_channels

        self.fusion = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(
                decoder_channels,
                output_channels,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: Sequence[Tensor],
        output_size: tuple[int, int],
    ) -> Tensor:
        """Project, resize and fuse multiscale encoder features."""
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature stages, received {len(features)}"
            )

        fusion_size = features[0].shape[-2:]
        projected_features = []

        for feature, projection in zip(
            features,
            self.projections,
            strict=True,
        ):
            projected = projection(feature)

            if projected.shape[-2:] != fusion_size:
                projected = functional.interpolate(
                    projected,
                    size=fusion_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected_features.append(projected)

        fused = self.fusion(torch.cat(projected_features, dim=1))

        return functional.interpolate(
            fused,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class PerceptionRTModel(nn.Module):
    """Shared MiT encoder with semantic, depth and uncertainty decoders."""

    def __init__(
        self,
        encoder: SegformerModel,
        *,
        number_of_classes: int,
        decoder_channels: int = 128,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()

        if number_of_classes <= 1:
            raise ValueError("Model requires at least two semantic classes")

        if decoder_channels <= 0:
            raise ValueError("Decoder channels must be positive")

        self.encoder = encoder
        encoder_channels = tuple(encoder.config.hidden_sizes)

        self.semantic_decoder = MultiScaleDecoder(
            encoder_channels,
            decoder_channels,
            number_of_classes,
            dropout_probability=dropout_probability,
        )
        self.depth_decoder = MultiScaleDecoder(
            encoder_channels,
            decoder_channels,
            1,
            dropout_probability=dropout_probability,
        )
        self.uncertainty_decoder = MultiScaleDecoder(
            encoder_channels,
            decoder_channels,
            1,
            dropout_probability=dropout_probability,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "nvidia/mit-b2",
        *,
        number_of_classes: int,
        decoder_channels: int = 128,
        dropout_probability: float = 0.1,
    ) -> "PerceptionRTModel":
        """Construct the multitask network with pretrained MiT weights."""
        encoder = SegformerModel.from_pretrained(checkpoint)

        return cls(
            encoder,
            number_of_classes=number_of_classes,
            decoder_channels=decoder_channels,
            dropout_probability=dropout_probability,
        )

    def forward(self, image: Tensor) -> PerceptionOutput:
        """Predict semantics, log-depth and depth uncertainty."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected image tensor shaped [B, 3, H, W], received {tuple(image.shape)}"
            )

        output_size = (image.shape[-2], image.shape[-1])
        encoder_output = self.encoder(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        features = encoder_output.hidden_states

        if features is None:
            raise RuntimeError("SegFormer encoder did not return hidden states")

        return PerceptionOutput(
            semantic_logits=self.semantic_decoder(features, output_size),
            log_depth=self.depth_decoder(features, output_size),
            depth_log_scale=self.uncertainty_decoder(
                features,
                output_size,
            ),
        )
