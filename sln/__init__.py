"""Soft Logit Normalization (SLN) loss.

Reference implementation accompanying the paper:

    "Soft Logit Normalization: A Geometry-Guided Loss for Class Separation
     and Variance Contraction".

The SLN loss rescales the logits before the softmax to

        f~ = f / (tau * ||f||_2 ** gamma)

and then applies the standard cross-entropy. It contains two endpoints:

    * gamma = 0, tau = 1   -> standard Cross-Entropy (CE)
    * gamma = 1            -> Logit Normalization (LN, Wei et al. 2022)
    * 0 < gamma < 1        -> Soft Logit Normalization (SLN)

Only the exponent ``gamma`` and the temperature ``tau`` are introduced.
"""
from .losses import (
    sln_logits,
    SoftLogitNormalization,
    SLNLoss,
    LogitNormalizationLoss,
    MSoftmaxLoss,
    LSoftmaxLoss,
)

from .cosine import CosineHead, training_logits

__all__ = [
    "sln_logits",
    "SoftLogitNormalization",
    "SLNLoss",
    "LogitNormalizationLoss",
    "MSoftmaxLoss",
    "LSoftmaxLoss",
    "CosineHead",
    "training_logits",
]
__version__ = "1.1.0"
