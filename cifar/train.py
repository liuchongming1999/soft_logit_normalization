"""CIFAR-10 / CIFAR-100 training with CE / LN / SLN losses.

Protocol follows He et al. (2016) and the paper's Appendix: 160 epochs,
batch size 128, SGD with momentum 0.9, learning rate divided by 10 at epochs
80 and 120, standard crop/flip augmentation.

Examples
--------
# SLN on CIFAR-10 (paper setting: gamma=0.7, tau=0.2, lr=0.01, wd=5e-3)
python train.py --dataset cifar10 --arch resnet20 --loss sln \
    --gamma 0.7 --tau 0.2 --lr 0.01 --wd 5e-3 --seed 0

# CE on CIFAR-10 (lr=0.1, wd=1e-4)
python train.py --dataset cifar10 --arch resnet20 --loss ce --lr 0.1 --wd 1e-4

# LN on CIFAR-100 (tau by the k^(-gamma/2) rule, ~= 0.063)
python train.py --dataset cifar100 --arch resnet20 --loss ln --tau 0.063 \
    --lr 0.1 --wd 1e-4

# SLN on CIFAR-100 (tau=0.08)
python train.py --dataset cifar100 --arch resnet20 --loss sln \
    --gamma 0.7 --tau 0.08 --lr 0.01 --wd 5e-3

# Combinations with generalization methods (paper comparison table):
python train.py --dataset cifar100 --arch resnet20 --loss ce --sam   # SAM + CE
python train.py --dataset cifar100 --arch resnet20 --loss sln --gamma 0.7 \
    --tau 0.08 --lr 0.01 --wd 5e-3 --sam                             # SAM + SLN
python train.py --dataset cifar100 --arch resnet20 --loss ce --mixup 1.0
python train.py --dataset cifar100 --arch resnet20 --loss ce --label-smoothing 0.1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sln import sln_logits, MSoftmaxLoss, LSoftmaxLoss  # noqa: E402
from models import resnet_cifar as ResNet  # noqa: E402

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


@torch.no_grad()
def evaluate(forward_fn, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def mixup(x, y, alpha):
    """Standard input mixup (Zhang et al. 2018). Returns mixed input and
    both target tensors with the mixing coefficient."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021).

    Thin wrapper that turns any ``torch.optim`` optimizer into SAM: each step
    performs an adversarial ascent (``ascent_step``) followed by the base
    optimizer update at the perturbed point (``descent_step``).
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, "SAM perturbation radius rho must be non-negative"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def descent_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None]), p=2)


def main():
    p = argparse.ArgumentParser(description="CIFAR-10/100 with CE/LN/SLN")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--arch", default="resnet20",
                   choices=["resnet20", "resnet32", "resnet56", "resnet110"])
    p.add_argument("--loss", default="sln",
                   choices=["ce", "ln", "sln", "msoftmax", "lsoftmax"])
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--m-margin", type=float, default=0.35, help="M-Softmax margin")
    p.add_argument("--l-margin", type=int, default=4, help="L-Softmax angular margin")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--no-amp", action="store_true")
    # generalization-method combinations (paper comparison table)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label smoothing for CE/LN/SLN (e.g. 0.1)")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="mixup alpha; 0 disables (e.g. 1.0)")
    p.add_argument("--sam", action="store_true",
                   help="use the SAM optimizer (Foret et al. 2021)")
    p.add_argument("--sam-rho", type=float, default=0.05)
    args = p.parse_args()

    tags = [args.loss]
    if args.sam:
        tags.append("sam")
    if args.mixup > 0:
        tags.append(f"mixup{args.mixup}")
    if args.label_smoothing > 0:
        tags.append(f"ls{args.label_smoothing}")
    run_dir = os.path.join(
        args.out_dir,
        f"{args.dataset}_{args.arch}_{'_'.join(tags)}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    mean, std, num_classes = CIFAR_STATS[args.dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds_cls = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
              else torchvision.datasets.CIFAR100)
    train_set = ds_cls(args.data_root, train=True, transform=train_tf, download=True)
    test_set = ds_cls(args.data_root, train=False, transform=test_tf, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    gamma = {"ce": 0.0, "ln": 1.0, "sln": args.gamma}.get(args.loss, 0.0)
    tau = 1.0 if args.loss == "ce" else args.tau

    model = getattr(ResNet, args.arch)(use_sln=False)
    if num_classes != 10:
        model.linear = nn.Linear(256, num_classes)

    # L-Softmax replaces the classifier head (operates on features).
    lsoftmax_head = None
    if args.loss == "lsoftmax":
        model.linear = nn.Identity()
        lsoftmax_head = LSoftmaxLoss(256, num_classes, margin=args.l_margin)

    model = model.to(device)
    if lsoftmax_head is not None:
        lsoftmax_head = lsoftmax_head.to(device)

    def criterion(feats_or_logits, target):
        if args.loss == "lsoftmax":
            return lsoftmax_head(feats_or_logits, target)
        if args.loss == "msoftmax":
            return m_criterion(feats_or_logits, target)
        return F.cross_entropy(sln_logits(feats_or_logits, gamma, tau), target,
                               label_smoothing=args.label_smoothing)

    m_criterion = MSoftmaxLoss(args.m_margin) if args.loss == "msoftmax" else None

    @torch.no_grad()
    def eval_logits(x):
        if args.loss == "lsoftmax":
            return lsoftmax_head.eval_logits(model.feature(x))
        return model(x)

    params = list(model.parameters())
    if lsoftmax_head is not None:
        params += list(lsoftmax_head.parameters())
    if args.sam:
        optimizer = SAM(params, torch.optim.SGD, rho=args.sam_rho, lr=args.lr,
                        momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.wd)
    use_amp = not args.no_amp and device.type == "cuda"

    def forward_loss(x, y_a, y_b, lam):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model.feature(x) if args.loss == "lsoftmax" else model(x)
            return lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)

    history = []
    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        if epoch in (80, 120):
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_a, y_b, lam = y, y, 1.0
            if args.mixup > 0:
                x, y_a, y_b, lam = mixup(x, y, args.mixup)
            if args.sam:
                loss = forward_loss(x, y_a, y_b, lam)
                optimizer.zero_grad()
                loss.backward()
                optimizer.ascent_step()
                loss = forward_loss(x, y_a, y_b, lam)
                optimizer.zero_grad()
                loss.backward()
                optimizer.descent_step()
            else:
                loss = forward_loss(x, y_a, y_b, lam)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        test_acc = evaluate(eval_logits, test_loader, device)
        best = max(best, test_acc)
        history.append({"epoch": epoch + 1, "test_acc": test_acc,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch+1}/{args.epochs} test_acc {test_acc:.4f}", flush=True)

    state = {"model": model.state_dict()}
    if lsoftmax_head is not None:
        state["lsoftmax_head"] = lsoftmax_head.state_dict()
    torch.save(state, os.path.join(run_dir, "model_final.pth"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    summary = {"best_test_acc": best, "final_test_acc": history[-1]["test_acc"],
               "wall_time_sec": time.time() - t0}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {run_dir} best {best:.4f} final {history[-1]['test_acc']:.4f}")


if __name__ == "__main__":
    main()
