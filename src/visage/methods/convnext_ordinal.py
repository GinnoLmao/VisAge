"""Pretrained ConvNeXt V2 Pico with a newly initialized ordinal head.

The timm dependency is imported only when this model is constructed. Use
``pretrained=False`` for offline architecture tests; production training
must explicitly record whether pretrained backbone weights were loaded.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn

from .ordinal import NUM_AGE_GROUPS, NUM_ORDINAL_LOGITS

MODEL_NAME = "convnextv2_pico.fcmae_ft_in1k"


class ConvNeXtOrdinal(nn.Module):
    """Wrap a timm backbone and replace its classifier with project logits.

    A frozen backbone stays in evaluation mode when ``train()`` is called,
    while the new head remains trainable. This makes frozen-feature and
    fine-tuned experiments use the same wrapper and checkpoint structure.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        pretrained_path: str | Path | None = None,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        head_type: str = "corn",
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if head_type not in {"corn", "softmax"}:
            raise ValueError("head_type must be 'corn' or 'softmax'.")
        if pretrained_path is not None and not pretrained:
            raise ValueError("pretrained_path requires pretrained=True.")
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "ConvNeXtOrdinal requires timm. Install the project's neural dependencies."
            ) from exc

        # num_classes=0 requests pooled features and discards ImageNet's head.
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained and pretrained_path is None,
            num_classes=0,
            global_pool="avg",
        )
        self.pretrained_path = (
            str(Path(pretrained_path).expanduser().resolve())
            if pretrained_path is not None
            else None
        )
        if self.pretrained_path is not None:
            self._load_pretrained_backbone(self.pretrained_path)
        self.num_features = self.backbone.num_features
        self.num_classes = NUM_AGE_GROUPS
        self.head_type = head_type
        self.pretrained_requested = pretrained
        self.dropout = nn.Dropout(dropout)
        output_count = NUM_ORDINAL_LOGITS if head_type == "corn" else NUM_AGE_GROUPS
        self.head = nn.Linear(self.num_features, output_count)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)
        self._backbone_trainable = True
        self.set_backbone_trainable(not freeze_backbone)

    def _load_pretrained_backbone(self, path: str) -> None:
        """Load a pinned safetensors artifact, excluding only ImageNet's head.

        The downloader is responsible for revision/checksum verification.
        Tensor names and shapes must match this exact timm backbone; prefix
        rewrites or silently ignored backbone parameters are not allowed.
        """
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading pretrained_path requires the safetensors package."
            ) from exc
        parameters = load_file(path, device="cpu")
        for key in ("head.fc.weight", "head.fc.bias"):
            parameters.pop(key, None)
        incompatible = self.backbone.load_state_dict(parameters, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Pretrained backbone parameters do not match "
                f"{MODEL_NAME}: missing={incompatible.missing_keys}; "
                f"unexpected={incompatible.unexpected_keys}."
            )

    @property
    def backbone_trainable(self) -> bool:
        return self._backbone_trainable

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze backbone weights, retaining a trainable new head."""
        self._backbone_trainable = bool(trainable)
        self.backbone.requires_grad_(self._backbone_trainable)
        self.head.requires_grad_(True)
        self.backbone.train(self.training and self._backbone_trainable)

    def train(self, mode: bool = True) -> ConvNeXtOrdinal:
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("inputs must have shape [batch, 3, height, width].")
        return self.backbone(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.dropout(self.forward_features(inputs)))


def create_convnext_ordinal(**kwargs: object) -> ConvNeXtOrdinal:
    """Construct the proposed modern method; defaults to pretrained weights."""
    return ConvNeXtOrdinal(**kwargs)


def set_backbone_trainable(model: ConvNeXtOrdinal, trainable: bool) -> None:
    """Expose the freeze/unfreeze operation to the shared training pipeline."""
    model.set_backbone_trainable(trainable)
