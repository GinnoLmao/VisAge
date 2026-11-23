"""VisAge's compact residual network, trained entirely from random weights.

Residual connections and squeeze-and-excitation are established components.
This module combines them for the project; it does not claim a novel method.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .ordinal import NUM_AGE_GROUPS, NUM_ORDINAL_LOGITS


class SqueezeExcitation(nn.Module):
    """Learn channel gates from each feature map's spatial averages."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive.")
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.gate(self.pool(inputs))


class ResidualSEBlock(nn.Module):
    """Two 3x3 convolutions, an optional SE gate, and a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = (
            SqueezeExcitation(out_channels, se_reduction) if use_se else nn.Identity()
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.attention(self.features(inputs))
        return self.activation(features + self.shortcut(inputs))


class VisAgeNet(nn.Module):
    """A four-stage, roughly 2.8M-parameter residual SE age-group model.

    ``head_type='corn'`` returns two conditional logits. ``'softmax'``
    returns three categorical logits for the controlled head ablation.
    Initialization uses PyTorch's current RNG; set its seed externally for
    reproducibility. This class never loads pretrained parameters.
    """

    def __init__(
        self,
        num_classes: int = NUM_AGE_GROUPS,
        *,
        widths: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.2,
        use_se: bool = True,
        se_reduction: int = 16,
        head_type: str = "corn",
    ) -> None:
        super().__init__()
        if num_classes != NUM_AGE_GROUPS:
            raise ValueError("VisAgeNet requires the project's three age groups.")
        if len(widths) != 4 or len(blocks_per_stage) != 4:
            raise ValueError("Exactly four stage widths and block counts are required.")
        if any(not isinstance(value, int) or value <= 0 for value in widths):
            raise ValueError("Stage widths must be positive integers.")
        if any(not isinstance(value, int) or value <= 0 for value in blocks_per_stage):
            raise ValueError("Stage block counts must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if head_type not in {"corn", "softmax"}:
            raise ValueError("head_type must be 'corn' or 'softmax'.")
        if se_reduction <= 0:
            raise ValueError("se_reduction must be positive.")

        self.num_classes = num_classes
        self.num_features = widths[-1]
        self.head_type = head_type
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, (out_channels, count) in enumerate(
            zip(widths, blocks_per_stage, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    ResidualSEBlock(
                        in_channels,
                        out_channels,
                        stride,
                        use_se=use_se,
                        se_reduction=se_reduction,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        output_count = NUM_ORDINAL_LOGITS if head_type == "corn" else num_classes
        self.head = nn.Linear(self.num_features, output_count)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("inputs must have shape [batch, 3, height, width].")
        return self.pool(self.stages(self.stem(inputs))).flatten(1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.dropout(self.forward_features(inputs)))


def create_visagenet(**kwargs: object) -> VisAgeNet:
    """Construct the randomly initialized project model."""
    return VisAgeNet(**kwargs)
