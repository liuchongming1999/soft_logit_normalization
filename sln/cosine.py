"""Class-level NormFace, ArcFace, CosFace and Circle objectives.

The CIFAR backbone exposes a 256-dimensional feature representation.
Label-dependent margins/adaptive weights are used during training only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

METHODS = {'normface', 'arcface', 'cosface', 'circle'}


class CosineHead(nn.Module):
    def __init__(self, backbone, classes, scale):
        super().__init__()
        backbone.linear = nn.Identity()
        self.backbone = backbone
        self.weight = nn.Parameter(torch.empty(classes, 256))
        nn.init.normal_(self.weight, std=0.01)
        self.scale = scale

    def feature(self, x):
        return self.backbone.feature(x)

    def forward(self, x):
        features = self.feature(x).float()
        with torch.autocast('cuda', enabled=False):
            cosine = F.linear(F.normalize(features, dim=1),
                              F.normalize(self.weight.float(), dim=1))
        return self.scale * cosine.clamp(-1 + 1e-7, 1 - 1e-7)


def training_logits(cosine, labels, method, margin, scale):
    cosine = cosine.float()
    rows = torch.arange(labels.numel(), device=labels.device)
    positive = cosine[rows, labels]
    if method == 'normface':
        return scale * cosine
    if method == 'cosface':
        target = positive - margin
    elif method == 'arcface':
        sine = (1 - positive.square()).clamp_min(1e-12).sqrt()
        raw = positive * math.cos(margin) - sine * math.sin(margin)
        target = torch.where(positive > math.cos(math.pi - margin), raw,
                             positive - math.sin(math.pi - margin) * margin)
    elif method == 'circle':
        # Sun et al., Eq. (6), one class-level positive and C-1 negatives.
        # Adaptive weights are detached, as in the paper's stated derivatives.
        alpha_n = (cosine.detach() + margin).clamp_min(0)
        adjusted = scale * alpha_n * (cosine - margin)
        alpha_p = (1 + margin - positive.detach()).clamp_min(0)
        adjusted[rows, labels] = scale * alpha_p * (positive - (1 - margin))
        return adjusted
    else:
        raise ValueError(method)
    adjusted = cosine.clone()
    adjusted[rows, labels] = target
    return scale * adjusted
