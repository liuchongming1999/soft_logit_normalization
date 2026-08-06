"""Soft Logit Normalization (SLN) loss.

Reference implementation accompanying the paper:

    "Soft Logit Normalization: Decoupling the Radial and Tangential
     Geometry of Logit Gradients" (NeurIPS 2026 submission).

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
)

__all__ = [
    "sln_logits",
    "SoftLogitNormalization",
    "SLNLoss",
    "LogitNormalizationLoss",
]
__version__ = "1.0.0"
