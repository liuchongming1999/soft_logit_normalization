"""Toy example: trainable-feature (free-feature) binary classification in 2D.

This reproduces the conceptual experiment of the paper. A *single linear
layer* with no hidden layers is used, and the 2D input points themselves are
treated as *trainable parameters* (the "trainable-feature" / unconstrained
feature model). We then watch where the features drift under the CE, LN and
SLN losses. The figure shows the per-point loss landscape (colour), the
decision boundary (black line), and the feature trajectories.

Run:

    python toy_example.py --loss sln --gamma 0.7 --tau 1.0 --out-dir figures
    python toy_example.py --loss ce
    python toy_example.py --loss ln

Outputs ``colorbar_<loss>.png`` (final state) and
``training_dynamics_<loss>.gif`` (feature drift over epochs).
"""
import argparse
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sln import sln_logits  # noqa: E402


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


class SimpleNN(nn.Module):
    """Single linear layer: 2 -> 2 (outputs raw logits)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


def generate_initial_data(num_samples):
    data = torch.randn(num_samples, 2)
    labels = torch.randint(0, 2, (num_samples,))
    return data, labels


def plot_state(data, labels, model, gamma, tau, save_path,
               xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), grid_size=500, colorbar=False):
    model.eval()
    data_np = data.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().ravel()
    # the stationary point x* = -W^{-1} b
    x_star = torch.linalg.solve(model.fc.weight, -model.fc.bias)

    xx, yy = np.meshgrid(np.linspace(*xlim, grid_size),
                         np.linspace(*ylim, grid_size))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        logits = sln_logits(model(grid), gamma, tau)
        pred = torch.argmax(logits, dim=1)
        zz = pred.cpu().numpy().reshape(xx.shape)
        per_point_loss = F.cross_entropy(logits, pred, reduction="none")
        loss_map = per_point_loss.cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(4, 4))
    plt.contourf(xx, yy, np.log1p(loss_map), levels=60, cmap="viridis_r", alpha=0.85)
    if colorbar:
        cbar = plt.colorbar(label="Log Loss", format=ticker.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=10)
    plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    plt.scatter(data_np[:, 0], data_np[:, 1], c=labels_np, cmap="coolwarm",
                edgecolor="k", s=50, zorder=3)
    plt.scatter(x_star[0].item(), x_star[1].item(), c="black", marker="x",
                s=100, label=r"$x^*$")
    plt.xlim(xlim); plt.ylim(ylim); plt.xticks([]); plt.yticks([])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="sln", choices=["ce", "ln", "sln"])
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--num-epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gamma = {"ce": 0.0, "ln": 1.0, "sln": args.gamma}[args.loss]
    tau = 1.0 if args.loss == "ce" else args.tau
    setup_seed(args.seed)

    model = SimpleNN()
    data, labels = generate_initial_data(args.num_samples)
    trainable_data = nn.Parameter(data.clone(), requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_data = optim.SGD([trainable_data], lr=args.lr)

    frames = []
    tmp = os.path.join(args.out_dir, "_tmp.png")
    for epoch in range(args.num_epochs):
        if epoch % args.snap_every == 0:
            plot_state(trainable_data, labels, model, gamma, tau, tmp)
            frames.append(imageio.imread(tmp))

        logits = sln_logits(model(trainable_data), gamma, tau)
        loss = criterion(logits, labels)
        optimizer_data.zero_grad()
        loss.backward()
        optimizer_data.step()

    plot_state(trainable_data, labels, model, gamma, tau,
               os.path.join(args.out_dir, f"colorbar_{args.loss}.png"), colorbar=True)
    imageio.mimsave(os.path.join(args.out_dir, f"training_dynamics_{args.loss}.gif"),
                    frames, duration=0.1)
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"[done] wrote colorbar_{args.loss}.png and training_dynamics_{args.loss}.gif")


if __name__ == "__main__":
    main()
