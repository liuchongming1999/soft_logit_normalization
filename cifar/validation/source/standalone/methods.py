"""Source-faithful margin objectives; all label-dependent changes are train-only."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
import resnet_cifar
from reference_train import AngularMarginNet, margin_training_logits, normalize_logits

METHODS = ['ce', 'ln', 'sln', 'arcface', 'cosface', 'msoftmax',
           'lsoftmax', 'sphereface', 'normface', 'circle']
COSINE_METHODS = {'arcface', 'cosface', 'normface', 'circle'}


def angular_psi(cosine, margin: int):
    """Liu et al.: (-1)^k cos(m theta) - 2k, via Chebyshev polynomials.

    Only the interval index k is detached. Polynomial and norm derivatives
    remain in the graph, including the L-Softmax weight-norm derivative.
    """
    if margin not in (1, 2, 3, 4):
        raise ValueError('integer margin must be 1, 2, 3 or 4')
    c = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
    a, b = torch.ones_like(c), c
    for _ in range(2, margin + 1):
        a, b = b, 2 * c * b - a
    with torch.no_grad():
        k = torch.floor(margin * torch.acos(c) / math.pi).clamp(max=margin - 1)
        sign = torch.where(k.remainder(2) == 0, 1., -1.)
    return sign * b - 2 * k


def angular_lambda(method, step):
    # Schedules from the authors' public CIFAR L-Softmax / SphereFace configs.
    if method == 'lsoftmax':
        return 1000. * (1. + 0.00002 * step) ** -45.
    if method == 'sphereface':
        return max(5., 1000. / (1. + 0.12 * step))
    return 0.


class MultiplicativeMarginNet(nn.Module):
    def __init__(self, backbone, method):
        super().__init__()
        backbone.linear = nn.Identity()
        self.backbone = backbone
        self.method = method
        self.weight = nn.Parameter(torch.empty(100, 256))
        if method == 'lsoftmax':
            nn.init.kaiming_normal_(self.weight, mode='fan_in')
        else:
            nn.init.xavier_uniform_(self.weight)

    def components(self, x):
        features = self.backbone.feature(x).float()
        with torch.autocast('cuda', enabled=False):
            weight = self.weight.float()
            if self.method == 'sphereface':
                weight = F.normalize(weight, dim=1)
            raw = F.linear(features, weight)
            norms = features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if self.method == 'lsoftmax':
                norms = norms * weight.norm(dim=1).clamp_min(1e-8)[None, :]
            cosine = (raw / norms).clamp(-1 + 1e-7, 1 - 1e-7)
        return raw, cosine, norms

    def forward(self, x):
        return self.components(x)[0]


def create_model(method, depth=20):
    backbone = getattr(resnet_cifar, f'resnet{depth}')(use_sln=False)
    if method in COSINE_METHODS:
        # Preserves the exact existing ArcFace/CosFace initialization and keys.
        return AngularMarginNet(backbone)
    if method in {'lsoftmax', 'sphereface'}:
        return MultiplicativeMarginNet(backbone, method)
    backbone.linear = nn.Linear(256, 100)
    return backbone


def circle_training_logits(cosine, labels, margin, scale):
    """Sun et al. Eq. 6, class-level single-positive Circle Loss.

    Stop-gradient adaptive weights implement the paper's pair-weighted
    derivatives. CE on these logits equals log(1 + sum exp(negative-positive)).
    """
    alpha_n = (cosine.detach() + margin).clamp_min(0)
    logits = scale * alpha_n * (cosine - margin)
    rows = torch.arange(labels.numel(), device=labels.device)
    positive = cosine[rows, labels]
    alpha_p = (1 + margin - positive.detach()).clamp_min(0)
    logits[rows, labels] = scale * alpha_p * (positive - (1 - margin))
    return logits


def train_logits(model, x, labels, cfg, step):
    method = cfg['method']
    rows = torch.arange(labels.numel(), device=labels.device)
    if method in COSINE_METHODS:
        cosine = model.cosine_logits(x)
        if method in {'arcface', 'cosface'}:
            adjusted = margin_training_logits(cosine, labels, method, cfg['margin'], cfg['scale'])
        elif method == 'circle':
            adjusted = circle_training_logits(cosine, labels, cfg['margin'], cfg['scale'])
        else:
            adjusted = cfg['scale'] * cosine
        return adjusted, cosine
    if method in {'lsoftmax', 'sphereface'}:
        raw, cosine, norms = model.components(x)
        target_norms = norms[:, 0] if norms.shape[1] == 1 else norms[rows, labels]
        phi = target_norms * angular_psi(cosine[rows, labels], int(cfg['margin']))
        lam = angular_lambda(method, step)
        adjusted = raw.clone()
        adjusted[rows, labels] = (lam * raw[rows, labels] + phi) / (1 + lam)
        return adjusted, raw
    logits = inference_logits(model, x, cfg)
    if method == 'msoftmax':
        adjusted = logits.clone()
        adjusted[rows, labels] -= cfg['margin']
        return adjusted, logits
    return logits, logits


def inference_logits(model, x, cfg):
    method = cfg['method']
    if method in COSINE_METHODS:
        return model(x, cfg['scale']).float()
    raw = model(x).float()
    if method in {'ce', 'msoftmax', 'lsoftmax', 'sphereface'}:
        return raw
    return normalize_logits(raw, 1. if method == 'ln' else cfg['gamma'], cfg['tau'])
