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
# Margin-based baselines compared in the paper
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
    """Large-Margin Softmax with the original unnormalized classifier.

    Consumes penultimate features and owns the bias-free classifier weight.
    The CIFAR annealing schedule is 1000 / (1 + 0.00002 * iteration) ** 45.
    At evaluation, call eval_logits(features) without the angular margin.
    The discrete angular branch is detached from automatic differentiation.
    """

    def __init__(self, in_features, out_features, margin=4,
                 lambda_base=1000.0, lambda_gamma=0.00002, lambda_power=45.0):
        super().__init__()
        self.margin = margin
        self.lambda_base = lambda_base
        self.lambda_gamma = lambda_gamma
        self.lambda_power = lambda_power
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        self.iter = 0
        # T_m coefficients in monomial basis (ascending order)
        coeffs = self._cheb_monomial(margin)
        self.register_buffer('cheb', torch.tensor(coeffs, dtype=torch.float32))

    @staticmethod
    def _cheb_monomial(m):
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

    def cos_m_theta(self, cos_theta):
        # evaluate T_m(cos_theta) via Horner in float32
        x = cos_theta.clamp(-1.0, 1.0)
        out = torch.zeros_like(x)
        for c in reversed(self.cheb.tolist()):
            out = out * x + c
        return out

    def forward(self, features, target):
        # features: (N, in_features); run in fp32 for numerical stability
        with torch.autocast('cuda', enabled=False):
            features = features.float()
            return self._forward_fp32(features, target)

    def _forward_fp32(self, features, target):
        w = self.weight                                 # (C, D); original L-Softmax
        x_norm = features.norm(p=2, dim=1)                # (N,)
        logits_w = F.linear(features, w)                  # |w_j||x| cos(theta_j)
        norm_product = x_norm.unsqueeze(1) * w.norm(p=2, dim=1).unsqueeze(0)
        cos_theta = (logits_w / norm_product.clamp(min=1e-8)).clamp(-1.0, 1.0)

        idx = torch.arange(features.size(0), device=features.device)
        cos_t_y = cos_theta[idx, target]
        # The segment index is a discrete lookup, not a differentiable angle.
        # Backpropagating acos through floor gives 0 * inf at cos(theta)=+/-1.
        # Keep the polynomial/norm gradients while freezing this branch choice.
        theta = torch.acos(cos_t_y.detach())
        k = (self.margin * theta / math.pi).floor().clamp(max=self.margin - 1)
        phi = ((-1.0) ** k) * self.cos_m_theta(cos_t_y) - 2.0 * k
        phi = phi * norm_product[idx, target]

        lam = self.lambda_base / ((1.0 + self.lambda_gamma * self.iter) ** self.lambda_power)
        f_y = (lam * logits_w[idx, target] + phi) / (1.0 + lam)

        logits = logits_w.clone()
        logits[idx, target] = f_y
        self.iter += 1
        return F.cross_entropy(logits, target)

    def eval_logits(self, features):
        """Margin-free logits with the unnormalized classifier weights."""
        return F.linear(features, self.weight)

    def _cos_m_theta(self, cos_theta):
        return self.cos_m_theta(cos_theta)

    def extra_repr(self):
        return (f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}, margin={self.margin}")
