"""Core SLN loss functions.

The functions here are deliberately minimal and framework-idiomatic so they
can be dropped into an existing training loop:

    >>> import torch
    >>> from sln import SLNLoss
    >>> criterion = SLNLoss(gamma=0.7, tau=0.2)
    >>> logits = model(images)                 # raw logits, any shape (N, K)
    >>> loss = criterion(logits, targets)      # scalar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def sln_logits(logits: torch.Tensor, gamma: float = 0.7, tau: float = 0.2) -> torch.Tensor:
    """Apply the SLN transform to raw logits.

    Computes ``logits / (tau * ||logits||_2 ** gamma)`` row-wise.

    Args:
        logits: Tensor of shape ``(N, K)`` (batch, classes).
        gamma:  Power of the logit norm. ``gamma=0`` recovers CE (up to the
                constant ``tau``), ``gamma=1`` recovers Logit Normalization.
        tau:    Temperature. The paper uses ``tau = O(k ** (-gamma / 2))``
                for ``k`` classes (e.g. 0.2 for CIFAR-10, 0.08 for CIFAR-100,
                0.2 with gamma=0.7 / 0.04 with gamma=1 for ImageNet).

    Returns:
        The rescaled logits, to be passed to ``F.cross_entropy`` or a softmax.
    """
    if gamma == 0.0 and tau == 1.0:
        return logits
    norms = logits.norm(p=2, dim=-1, keepdim=True)
    return logits / (((norms + _EPS) ** gamma) * tau)


class SoftLogitNormalization(nn.Module):
    """SLN logit transform as an ``nn.Module`` (returns transformed logits)."""

    def __init__(self, gamma: float = 0.7, tau: float = 0.2):
        super().__init__()
        self.gamma = float(gamma)
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return sln_logits(logits, self.gamma, self.tau)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, tau={self.tau}"


class SLNLoss(nn.Module):
    """Soft Logit Normalization cross-entropy loss.

    Args:
        gamma: Power of the logit norm (0.7 in the paper).
        tau:   Temperature (0.2 for CIFAR-10 / GLUE, 0.08 for CIFAR-100).
        label_smoothing: Optional label smoothing passed to cross_entropy.
    """

    def __init__(self, gamma: float = 0.7, tau: float = 0.2, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = sln_logits(logits, self.gamma, self.tau)
        return F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, tau={self.tau}, label_smoothing={self.label_smoothing}"


class LogitNormalizationLoss(SLNLoss):
    """Logit Normalization (Wei et al. 2022) — the ``gamma=1`` endpoint."""

    def __init__(self, tau: float = 0.04, label_smoothing: float = 0.0):
        super().__init__(gamma=1.0, tau=tau, label_smoothing=label_smoothing)
