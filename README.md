# Soft Logit Normalization (SLN)

Official code for the paper **"Soft Logit Normalization: A Geometry-Guided
Loss for Class Separation and Variance Contraction"** (Chongming Liu,
Songting Li, Douglas Zhou).

SLN is a minimal, drop-in loss modification. It rescales the logits before the
softmax to

$$
\tilde{\mathbf f}=\frac{\mathbf f}{\tau\,\lVert\mathbf f\rVert_2^{\gamma}},
$$

and then applies the standard cross-entropy. The single new exponent
$\gamma$ interpolates between two well-known objectives:

| $\gamma$ | objective |
|---|---|
| $\gamma=0,\ \tau=1$ | standard Cross-Entropy (CE) |
| $\gamma=1$ | Logit Normalization (LN, [Wei et al. 2022](https://arxiv.org/abs/2203.11772)) |
| $0<\gamma<1$ | **Soft Logit Normalization (SLN)** |

Geometrically, the Jacobian of the SLN logit transform has a *tangential*
eigenvalue $1/(\tau\lVert\mathbf f\rVert_2^\gamma)$ and a *radial* eigenvalue
$(1-\gamma)/(\tau\lVert\mathbf f\rVert_2^\gamma)$, whose ratio is exactly
$1-\gamma$ — independent of the datapoint. LN ($\gamma=1$) deletes the radial
component entirely; SLN restores a controlled fraction $1-\gamma$ of it.

<p align="center">
  <img src="assets/gradient_sln.png" width="90%">
</p>

*Trainable-feature toy example: a linear model on 2D points whose inputs are
themselves optimized. Features drift along the loss-induced directions
(here SLN). See [`toy_example/`](toy_example/).*

---

## Installation

```bash
git clone https://github.com/liuchongming1999/soft_logit_normalization.git
cd soft_logit_normalization
pip install -r requirements.txt
```

The core loss has **no other dependency than PyTorch** and can also be used
standalone (just copy [`sln/losses.py`](sln/losses.py)).

## Use SLN in your own training

```python
import torch
from sln import SLNLoss

criterion = SLNLoss(gamma=0.7, tau=0.2)   # tau = O(k^(-gamma/2)), k = #classes

for images, targets in train_loader:
    logits = model(images)                # raw logits
    loss = criterion(logits, targets)     # SLN cross-entropy
    loss.backward()
    optimizer.step()
```

If you already compute logits and prefer to transform them yourself:

```python
from sln import sln_logits
loss = torch.nn.functional.cross_entropy(sln_logits(logits, gamma=0.7, tau=0.2), targets)
```

`LogitNormalizationLoss(tau=0.04)` is provided as the $\gamma=1$ endpoint.

The same [`sln/losses.py`](sln/losses.py) also ships the margin-based
baselines compared in the paper, so those comparison rows can be reproduced:

```python
from sln import MSoftmaxLoss, LSoftmaxLoss

m_criterion = MSoftmaxLoss(margin=0.35)          # M-Softmax (logit-level, like CE)
head = LSoftmaxLoss(in_features=256, out_features=10, margin=4)  # L-Softmax
loss = head(backbone.feature(images), targets)   # replaces the final linear layer
```

> **Choosing $\tau$.** The paper uses $\tau = O(k^{-\gamma/2})$ for $k$
> classes so that the transformed logits start in a well-conditioned range:
> $\tau=0.2$ for CIFAR-10 ($k=10$), $\tau\approx0.08$ for CIFAR-100 ($k=100$),
> and $\tau=0.2$ for ImageNet-1K and GLUE; on ImageNet the LN endpoint
> ($\gamma=1$) uses $\tau=0.04$. We use $\gamma=0.7$ throughout, selected on
> CIFAR-10 and transferred unchanged.

---

## Repository structure

```
sln/            # the core SLN loss (importable, dependency-free except torch)
toy_example/    # 2D trainable-feature toy experiment + figures/gifs
cifar/          # CIFAR-10 / CIFAR-100 training (CE / LN / SLN, ResNet-20..110)
imagenet/       # ImageNet-1K training (timm-based, ResNet / ViT)
glue/           # GLUE fine-tuning (BERT / DeBERTa, CE / LN / SLN)
assets/         # figures used in this README
```

Each sub-directory has its own README with exact commands.

---

## Results

### CIFAR-10 / CIFAR-100 (ResNet, best-test accuracy, mean ± std over 10 runs)

| Model | #Param | Loss | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| ResNet-20 | 0.27M | CE | 91.81±0.15 | 67.26±0.18 |
| | | LN | 92.05±0.10 | 67.33±0.16 |
| | | **SLN** | **92.58±0.08** | **69.75±0.25** |
| ResNet-32 | 0.46M | CE | 92.40±0.24 | 68.62±0.33 |
| | | LN | 92.45±0.22 | 68.92±0.42 |
| | | **SLN** | **93.19±0.17** | **71.41±0.21** |
| ResNet-56 | 0.86M | CE | 93.07±0.17 | 70.19±0.22 |
| | | LN | 93.16±0.21 | 70.32±0.19 |
| | | **SLN** | **93.78±0.18** | **72.76±0.23** |
| ResNet-110 | 1.73M | CE | 93.64±0.16 | 70.84±0.31 |
| | | LN | 93.67±0.20 | 71.03±0.20 |
| | | **SLN** | **94.10±0.09** | **73.64±0.29** |

The advantage is particularly pronounced on CIFAR-100, where SLN exceeds the
stronger of CE and LN by 2.42–2.61 pp across all four depths. A ResNet-20
trained with SLN matches a ResNet-56 (~3× parameters) trained with CE/LN;
likewise ResNet-32+SLN approaches or exceeds ResNet-110+CE/LN. Reproduce with
[`cifar/train.py`](cifar/).

### Comparison with generalization methods (CIFAR-100, ResNet-20)

| Method | Best-test accuracy |
|---|---|
| CE | 67.26±0.18 |
| LN | 67.33±0.16 |
| Label smoothing + CE | 67.89±0.21 |
| Mixup + CE | 68.61±0.27 |
| SAM + CE | 69.04±0.27 |
| SLN | 69.75±0.25 |
| Label smoothing + SLN | 68.31±0.34 |
| Mixup + SLN | 69.93±0.17 |
| **SAM + SLN** | **70.12±0.18** |
| M-Softmax | 67.37±0.35 |
| L-Softmax | 67.75±0.23 |

Standalone SLN exceeds every evaluated alternative that does not use SLN
(including SAM+CE); Mixup and SAM further improve SLN, while label smoothing
does not help under the evaluated search. The CE/LN/SLN rows reuse the
ten-run values above; the other rows use five seeds. Reproduce with the
`--sam`, `--mixup` and `--label-smoothing` flags of [`cifar/train.py`](cifar/).

<p align="center">
  <img src="assets/gamma_test_accuracy.png" width="60%">
</p>

*Effect of the exponent $\gamma$ on CIFAR-10 (ResNet-20) best-test accuracy;
the sweep peaks at the theory-guided default $\gamma=0.7$.*

### ImageNet-1K (ResNet, single run)

| Model | #Param | Loss | Top-1 | Top-5 |
|---|---|---|---|---|
| ResNet-50 | 26M | CE | 76.8 | 93.0 |
| | | LN | 78.1 | 93.8 |
| | | **SLN** | **79.2** | **94.6** |
| ResNet-101 | 45M | CE | 78.6 | 93.9 |
| | | LN | 79.4 | 93.0 |
| | | **SLN** | **80.5** | **95.3** |
| ResNet-152 | 60M | CE | 78.9 | 94.1 |
| | | LN | 79.9 | 93.2 |
| | | **SLN** | **81.0** | **95.5** |

SLN attains the highest top-1 and top-5 accuracy at every evaluated depth;
ResNet-50+SLN (79.2%) already surpasses the much larger ResNet-152+CE
(78.9%). Reproduce with [`imagenet/`](imagenet/).

<p align="center">
  <img src="assets/imagenet_result.png" width="90%">
</p>

### GLUE (BERT, best development set)

| Model | Loss | MNLI (m/mm) | QQP | QNLI | SST-2 | CoLA | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| BERT-BASE | CE | 84.5/83.5 | 88.1 | 90.9 | 93.2 | 61.0 | 90.9 | 68.6 | 82.6 |
| | LN | 85.0/84.3 | 88.4 | 91.6 | 92.8 | 61.3 | 89.3 | 68.8 | 82.7 |
| | **SLN** | 85.0/84.9 | 88.5 | 91.7 | 93.5 | 63.9 | 91.4 | 71.7 | **83.8** |
| BERT-LARGE | CE | 86.2/85.5 | 88.6 | 92.4 | 93.7 | 62.3 | 91.2 | 72.3 | 84.0 |
| | LN | 86.5/86.1 | 88.8 | 92.6 | 93.2 | 62.7 | 90.2 | 73.9 | 84.2 |
| | **SLN** | 86.8/86.6 | 88.9 | 92.7 | 94.0 | 64.7 | 91.5 | 74.7 | **85.0** |

BERT-BASE+SLN reaches the average of BERT-LARGE (~3× parameters) trained with
CE/LN. Reproduce with [`glue/`](glue/).

### Representation geometry (CIFAR-10, ResNet-20, epoch 160, 10 seeds)

| Loss | NC1 ↓ | Intra/inter ratio ↓ |
|---|---|---|
| CE | 1.552±0.073 | 0.833±0.020 |
| LN | 0.656±0.029 | 0.500±0.009 |
| **SLN** | **0.386±0.010** | **0.399±0.006** |

<p align="center">
  <img src="assets/intra_inter_dynamics.png" width="45%">
  <img src="assets/nc1_dynamics.png" width="45%">
</p>

*Evolution of the representation metrics over training (mean ± std over ten
seeds). SLN produces the tightest within-class clouds relative to class-mean
separation throughout training.*

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{liu2026soft,
  title  = {Soft Logit Normalization: A Geometry-Guided Loss for Class
            Separation and Variance Contraction},
  author = {Liu, Chongming and Li, Songting and Zhou, Douglas},
  year   = {2026}
}
```

## License

[MIT](LICENSE) © 2026 Chongming Liu (Truman)
