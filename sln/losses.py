"""Core SLN loss functions.

The functions here are deliberately minimal and framework-idiomatic so they
can be dropped into an existing training loop:

    >>> import torch
    >>> from sln import SLNLoss
    >>> criterion = SLNLoss(gamma=0.7, tau=0.2)
    >>> logits = model(images)                 # raw logits, any shape (N, K)
    >>> loss = criterion(logits, targets)      # scalar
"""
import math

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


# --------------------------------------------------------------------------
# Margin-based baselines compared in the paper (Table 1)
# --------------------------------------------------------------------------


class MSoftmaxLoss(nn.Module):
    """Ensemble soft-margin softmax (M-Softmax, Wang et al. 2018).

    Subtracts a fixed margin ``m`` from the ground-truth logit and applies
    cross-entropy::

        L = -log( exp(f_y - m) / (exp(f_y - m) + sum_{j != y} exp(f_j)) )

    This is a plain logit-level loss (takes raw logits, like CE/SLN).
    """

    def __init__(self, margin: float = 0.35):
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), target] -= self.margin
        return F.cross_entropy(adjusted, target)

    def extra_repr(self) -> str:
        return f"margin={self.margin}"


class LSoftmaxLoss(nn.Module):
    """Large-Margin Softmax (L-Softmax, Liu et al. 2016).

    Unlike CE/LN/SLN/M-Softmax, L-Softmax replaces the **final linear layer**:
    it holds its own classifier weight and consumes the penultimate *features*
    (not logits), imposing an angular margin between the feature and its class
    weight. Following the original CIFAR experiments, a lambda-annealing
    schedule is used to interpolate between the standard and margin logits.

    Args:
        in_features:  dimension of the penultimate feature.
        out_features: number of classes.
        margin:       angular margin ``m`` (4 in the original CIFAR setting).
        lambda_base / lambda_gamma / lambda_power: annealing schedule.

    Example (use as the classifier head)::

        head = LSoftmaxLoss(256, 10)
        feats = backbone.feature(images)     # penultimate features
        loss = head(feats, targets)
        # evaluation logits (no margin, normalized weights):
        eval_logits = head.eval_logits(feats)
    """

    def __init__(self, in_features: int, out_features: int, margin: int = 4,
                 lambda_base: float = 1500.0, lambda_gamma: float = 0.00012,
                 lambda_power: float = 1.0):
        super().__init__()
        self.margin = int(margin)
        self.lambda_base = float(lambda_base)
        self.lambda_gamma = float(lambda_gamma)
        self.lambda_power = float(lambda_power)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        self.iter = 0
        # Chebyshev coefficients of T_m(x) in the monomial basis (ascending).
        self.register_buffer("cheb", torch.tensor(self._cheb_monomial(margin),
                                                  dtype=torch.float32))

    @staticmethod
    def _cheb_monomial(m: int):
        if m == 0:
            return [1.0]
        t_prev, t_cur = [1.0], [0.0, 1.0]
        for _ in range(2, m + 1):
            nxt = [0.0] * (len(t_cur) + 1)
            for i, c in enumerate(t_cur):
                nxt[i + 1] += 2.0 * c
            for i, c in enumerate(t_prev):
                nxt[i] -= c
            t_prev, t_cur = t_cur, nxt
        return t_cur

    def _cos_m_theta(self, cos_theta: torch.Tensor) -> torch.Tensor:
        x = cos_theta.clamp(-1.0, 1.0)
        out = torch.zeros_like(x)
        for c in reversed(self.cheb.tolist()):
            out = out * x + c
        return out

    def eval_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Margin-free logits with normalized weights (for evaluation)."""
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(features, w)

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # run in fp32 for numerical stability of acos / Chebyshev
        with torch.autocast("cuda", enabled=False):
            features = features.float()
            w = F.normalize(self.weight, p=2, dim=1)          # (C, D)
            x_norm = features.norm(p=2, dim=1)                # (N,)
            logits_w = F.linear(features, w)                  # (N, C) = |x| cos
            cos_theta = (logits_w / x_norm.clamp(min=_EPS).unsqueeze(1)).clamp(-1.0, 1.0)

            idx = torch.arange(features.size(0), device=features.device)
            cos_t_y = cos_theta[idx, target]
            theta = torch.acos(cos_t_y)
            k = (self.margin * theta / math.pi).floor()
            phi = ((-1.0) ** k) * self._cos_m_theta(cos_t_y) - 2.0 * k
            phi = phi * x_norm

            lam = self.lambda_base / ((1.0 + self.lambda_gamma * self.iter) ** self.lambda_power)
            f_y = (lam * logits_w[idx, target] + phi) / (1.0 + lam)

            logits = logits_w.clone()
            logits[idx, target] = f_y
            self.iter += 1
            return F.cross_entropy(logits, target)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, margin={self.margin}"
