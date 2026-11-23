"""Strict pretrained loading and architecture-independent three-class heads."""
from pathlib import Path
import torch
from torch import nn
from .model_registry import REGISTRY
from .runtime import sha256_file


class PretrainedAgeModel(nn.Module):
    head_type = 'softmax'

    def __init__(self, method, *, weights=None, provenance=None, gradient_checkpointing=False):
        super().__init__()
        import timm
        self.method = method
        self.spec = REGISTRY[method]
        # Never use network loading here. Training provides verified local weights;
        # inference reconstructs the architecture and loads the complete artifact.
        self.backbone = timm.create_model(self.spec.checkpoint, pretrained=False, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 3)
        nn.init.normal_(self.head.weight, 0, .01)
        nn.init.zeros_(self.head.bias)
        self.pretrained_loaded = False
        self._trainable = True
        if weights is not None:
            if not provenance or provenance['repository'] != f'timm/{self.spec.checkpoint}' or sha256_file(Path(weights)) != provenance['sha256']:
                raise ValueError('Pretrained file does not match pinned provenance.')
            from safetensors.torch import load_file
            state = load_file(str(weights), device='cpu')
            excluded = {self.spec.classifier + '.weight', self.spec.classifier + '.bias'}
            if not excluded.issubset(state):
                raise ValueError('Expected published ImageNet classifier is missing.')
            state = {key: value for key, value in state.items() if key not in excluded}
            # Older Swin exports serialize deterministic, nonpersistent masks.
            # Verify their full values against the constructed256px architecture;
            # never silently discard a learned or mismatched tensor.
            expected_keys = set(self.backbone.state_dict())
            self.verified_attention_masks = []
            for key in list(state):
                if key not in expected_keys and key.endswith('.attn_mask'):
                    generated = self.backbone.get_buffer(key)
                    if generated is None or not torch.equal(state[key], generated):
                        raise ValueError(f'Published attention mask differs from generated mask: {key}')
                    self.verified_attention_masks.append(key)
                    del state[key]
            # Full strict loading of every learned weight and persistent buffer.
            self.backbone.load_state_dict(state, strict=True)
            if any(not torch.equal(value, self.backbone.state_dict()[key]) for key, value in state.items()):
                raise ValueError('Backbone values differ from the published checkpoint.')
            self.pretrained_loaded = True
        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing(True)

    def set_backbone_trainable(self, value):
        self._trainable = bool(value)
        self.backbone.requires_grad_(self._trainable)
        self.backbone.train(self.training and self._trainable)
        self.head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.train(mode and self._trainable)
        return self

    def forward(self, inputs):
        return self.head(self.backbone(inputs))
