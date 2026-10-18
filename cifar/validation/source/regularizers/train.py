"""CIFAR-100 reviewer comparison: validation-only model selection.

The official test split is deliberately absent from this trainer. It is read only
by evaluate_selected.py after the complete validation grid has finished.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms

import resnet_cifar


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
METHODS = {"ce", "sln", "ls_ce", "ls_sln", "mixup_ce", "mixup_sln", "sam_ce", "sam_sln"}


def normalize_logits(logits: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
    logits = logits.float()
    if gamma == 0 and tau == 1:
        return logits
    return logits / (tau * logits.norm(dim=1, keepdim=True).clamp_min(1e-8).pow(gamma))


class AngularMarginNet(nn.Module):
    """ResNet-20 features + normalized feature/weight head for ArcFace/CosFace."""

    def __init__(self, backbone: nn.Module, classes: int = 100):
        super().__init__()
        backbone.linear = nn.Identity()
        self.backbone = backbone
        self.weight = nn.Parameter(torch.empty(classes, 256))
        nn.init.normal_(self.weight, std=0.01)

    def cosine_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.feature(x).float()
        with torch.autocast("cuda", enabled=False):
            cosine = F.linear(F.normalize(features, dim=1),
                              F.normalize(self.weight.float(), dim=1))
        return cosine.clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * self.cosine_logits(x)


def margin_training_logits(cosine: torch.Tensor, labels: torch.Tensor, method: str, margin: float, scale: float) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False):
        cosine = cosine.float()
        indices = torch.arange(labels.numel(), device=labels.device)
        target = cosine[indices, labels]
        if method == "cosface":
            phi = target - margin
        elif method == "arcface":
            # Standard ArcFace non-easy-margin monotonic branch.
            sine = torch.sqrt((1.0 - target.square()).clamp_min(1e-12))
            phi_raw = target * math.cos(margin) - sine * math.sin(margin)
            threshold = math.cos(math.pi - margin)
            phi = torch.where(target > threshold, phi_raw, target - math.sin(math.pi - margin) * margin)
        else:
            raise ValueError(method)
        adjusted = cosine.clone()
        adjusted[indices, labels] = phi
        return scale * adjusted


def create_model(method: str) -> nn.Module:
    backbone = resnet_cifar.resnet20(use_sln=False)
    if method in {"arcface", "cosface"}:
        return AngularMarginNet(backbone)
    backbone.linear = nn.Linear(256, 100)
    return backbone


def train_val_split(targets: list[int]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(targets)
    rng = np.random.RandomState(12345)
    train_parts, val_parts = [], []
    for c in range(100):
        idx = np.where(labels == c)[0]
        assert len(idx) == 500
        rng.shuffle(idx)
        val_parts.append(idx[:50])
        train_parts.append(idx[50:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert not np.intersect1d(train_idx, val_idx).size
    return train_idx, val_idx


def loaders(data_root: str, batch_size: int, seed: int, workers: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_full = CIFAR100(root=data_root, train=True, transform=train_tf, download=False)
    val_full = CIFAR100(root=data_root, train=True, transform=eval_tf, download=False)
    train_idx, val_idx = train_val_split(train_full.targets)
    generator = torch.Generator().manual_seed(100000 + seed)
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                              shuffle=True, generator=generator, drop_last=True,
                              num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=512,
                            shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def inference_logits(model: nn.Module, x: torch.Tensor, config: dict) -> torch.Tensor:
    method = config["method"].split("_")[-1]
    if method in {"arcface", "cosface"}:
        return model(x, config["scale"]).float()
    raw = model(x)
    gamma = 0.0 if method == "ce" else (1.0 if method == "ln" else config["gamma"])
    tau = 1.0 if method == "ce" else config["tau"]
    return normalize_logits(raw, gamma, tau)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, config: dict) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        y = y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = inference_logits(model, x, config)
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        total += len(y)
    return correct / total, loss_sum / total


def save_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temp, path)


def training_loss(model, x, y, cfg, lam=None, other_y=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = inference_logits(model, x, cfg)
        eps = cfg.get("ls_eps", 0.0)
        loss = F.cross_entropy(logits, y, label_smoothing=eps)
        if lam is not None:
            loss = lam * loss + (1 - lam) * F.cross_entropy(logits, other_y, label_smoothing=eps)
    return loss, logits


def optimization_step(model, optimizer, x, y, cfg):
    lam = other_y = None
    if cfg.get("mixup_alpha", 0.0) > 0:
        alpha = cfg["mixup_alpha"]
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[perm]
        other_y = y[perm]
    loss, logits = training_loss(model, x, y, cfg, lam, other_y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite first-pass loss")
    loss.backward()
    rho = cfg.get("sam_rho", 0.0)
    if rho > 0:
        with torch.no_grad():
            params = [p for p in model.parameters() if p.grad is not None]
            grad_norm = torch.stack([p.grad.norm(p=2) for p in params]).norm(p=2)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite SAM gradient norm")
            # Store original parameters, so restoration is exact rather than p+=e; p-=e.
            originals = [(p, p.detach().clone()) for p in params]
            for p in params:
                p.add_(p.grad * (rho / grad_norm.clamp_min(1e-12)))
        optimizer.zero_grad(set_to_none=True)
        # Use current-batch statistics on both passes; update running buffers only once.
        bn_modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        tracking = [m.track_running_stats for m in bn_modules]
        try:
            for m in bn_modules:
                m.track_running_stats = False
            second_loss, _ = training_loss(model, x, y, cfg, lam, other_y)
            if not torch.isfinite(second_loss):
                raise FloatingPointError("Non-finite SAM perturbed loss")
            second_loss.backward()
        finally:
            with torch.no_grad():
                for param, original in originals:
                    param.copy_(original)
            for m, value in zip(bn_modules, tracking):
                m.track_running_stats = value
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Mixup training accuracy is label-weighted; validation is ordinary accuracy.
    correct = (logits.detach().argmax(1) == y).float().sum().item()
    if lam is not None:
        correct = lam * correct + (1 - lam) * (logits.detach().argmax(1) == other_y).float().sum().item()
    return loss.detach().item(), correct


def audit_initial_state(out, model, train_loader, val_loader):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    train_idx = np.asarray(train_loader.dataset.indices, dtype=np.int64)
    val_idx = np.asarray(val_loader.dataset.indices, dtype=np.int64)
    np.savez_compressed(out / "split_indices.npz", train=train_idx, val=val_idx)
    save_json(out / "initial_audit.json", {
        "initial_state_sha256": digest.hexdigest(),
        "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "training_examples": len(train_idx), "validation_examples": len(val_idx),
        "data_generator_seed": 100000 + int(json.loads((out / "config.json").read_text())["seed"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["method"] not in METHODS:
        raise ValueError(cfg["method"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", cfg)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = loaders(cfg["data_root"], cfg["batch_size"], seed, cfg["workers"])
    model = create_model(cfg["method"]).cuda().to(memory_format=torch.channels_last)
    audit_initial_state(out, model, train_loader, val_loader)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    history_path = out / "history.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_nll", "epoch_seconds"]
    with history_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best_acc, best_epoch = -1.0, 0
    start = time.time()
    method = cfg["method"]
    for epoch in range(1, cfg["epochs"] + 1):
        if epoch in (81, 121):
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        model.train()
        ep_loss = ep_correct = ep_total = 0
        epoch_start = time.time()
        for x, y in train_loader:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            loss_value, correct = optimization_step(model, optimizer, x, y, cfg)
            ep_loss += loss_value * len(y)
            ep_correct += correct
            ep_total += len(y)
        val_acc, val_nll = evaluate(model, val_loader, cfg)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": val_acc}, out / "best.pth")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": ep_loss / ep_total, "train_acc": ep_correct / ep_total,
               "val_acc": val_acc, "val_nll": val_nll,
               "epoch_seconds": time.time() - epoch_start}
        with history_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        save_json(out / "status.json", {"epoch": epoch, "best_epoch": best_epoch,
                                        "best_val_acc": best_acc, "last_val_acc": val_acc,
                                        "method": method})
        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, out / "last.pth")
    save_json(out / "summary.json", {"method": method, "seed": seed,
                                     "best_epoch": best_epoch, "best_val_acc": best_acc,
                                     "final_val_acc": val_acc,
                                     "duration_seconds": time.time() - start})
    print("COMPLETE", out, "best_val", best_acc, "epoch", best_epoch, flush=True)


if __name__ == "__main__":
    main()
