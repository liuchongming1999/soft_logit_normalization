"""Normalized heads used by the audited CIFAR trainer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sln.cosine import CosineHead, METHODS, training_logits
__all__ = ["CosineHead", "METHODS", "training_logits"]
