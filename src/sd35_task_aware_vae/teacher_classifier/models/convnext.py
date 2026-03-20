from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Large_Weights, convnext_large


class ConvNeXtLargeTeacher(nn.Module):
    """ConvNeXt-Large teacher with explicit feature / embedding APIs.

    This wrapper preserves the familiar torchvision attributes
    (`features`, `avgpool`, `classifier`) so legacy training/eval scripts keep
    working, while adding task-aware helper methods used by the SD3.5 codepath.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_large(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

    def forward_features(self, x: torch.Tensor, stage: int | str | None = None):
        """Return intermediate or final spatial features.

        stage:
          - None: final feature map
          - int : stage index from the ConvNeXt `features` sequential
          - "all": list of all intermediate feature maps
        """
        feats: list[torch.Tensor] = []
        h = x
        for block in self.features:
            h = block(h)
            feats.append(h)

        if stage is None:
            return feats[-1]
        if stage == "all":
            return feats
        if isinstance(stage, int):
            if stage < 0:
                stage = len(feats) + stage
            if stage < 0 or stage >= len(feats):
                raise IndexError(f"stage out of range: {stage}")
            return feats[stage]
        raise ValueError(f"Unsupported stage specifier: {stage}")

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.avgpool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embedding(x)
        return self.classifier(emb)


def build_convnext_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    return ConvNeXtLargeTeacher(num_classes=num_classes, pretrained=pretrained)
