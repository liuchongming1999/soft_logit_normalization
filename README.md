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

The package also provides M-Softmax, the original unnormalized L-Softmax
head, and class-level NormFace/ArcFace/CosFace/Circle implementations. The
audited CIFAR comparison uses [`cifar/train_besttest.py`](cifar/train_besttest.py):

```python
from sln import MSoftmaxLoss, LSoftmaxLoss

m_criterion = MSoftmaxLoss(margin=1.0)          # M-Softmax (logit-level, like CE)
head = LSoftmaxLoss(in_features=256, out_features=10, margin=4)  # L-Softmax
loss = head(backbone.feature(images), targets)   # replaces the final linear layer
```

> **Choosing $\tau$.** The paper uses $\tau = O(k^{-\gamma/2})$ for $k$
> classes so that the transformed logits start in a well-conditioned range:
> $\tau=0.2$ for CIFAR-10 ($k=10$), $\tau\approx0.08$ for CIFAR-100 ($k=100$),
> and $\tau=0.2$ for ImageNet-1K and GLUE; on ImageNet the LN endpoint
> ($\gamma=1$) uses $\tau=0.04$. $\gamma=0.7$ is a practical default informed by the local contraction balance and
empirical performance; the accuracy-maximizing exponent can depend on the task and recipe.

---

## Repository structure

```
sln/            # the core SLN loss (importable, dependency-free except torch)
toy_example/    # 2D trainable-feature toy experiment + figures/gifs
cifar/          # CIFAR training, audited best-test grids, baselines and results
theory/         # independent multiclass local-dynamics verification
imagenet/       # ImageNet-1K training (timm-based, ResNet / ViT)
glue/           # GLUE fine-tuning (BERT / DeBERTa, CE / LN / SLN)
assets/         # figures used in this README
```

Each sub-directory has its own README with exact commands.

---

## Multiclass dynamics

The multiclass extension resolves radial, class-tangent and common-logit
directions under balanced, locally concentrated populations with
regular-simplex class means and direct logit gradient flow. Independent
float64 checks of the Hessian blocks, variance trajectories and local
covariance dynamics are provided in [`theory/`](theory/).

```bash
python theory/verify_multiclass.py
```

## Results

### CIFAR-10 / CIFAR-100

Best-test accuracy (%). CIFAR-10 retains the manuscript's ten-run results.
CIFAR-100 uses five seeds at one configuration selected by their mean
best-test accuracy from nine configurations per method and backbone depth.
The reported uncertainty is the sample standard deviation.

| Model | Params (C10/C100) | Loss | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|
| ResNet-20 | 0.27/0.30M | CE | 91.81±0.15 | 69.14±0.22 |
|  |  | LN | 92.05±0.10 | 69.26±0.21 |
|  |  | SLN | **92.58±0.08** | **69.55±0.48** |
| ResNet-32 | 0.47/0.49M | CE | 92.40±0.24 | 70.95±0.34 |
|  |  | LN | 92.45±0.22 | 70.85±0.21 |
|  |  | SLN | **93.19±0.17** | **71.40±0.25** |
| ResNet-56 | 0.86/0.88M | CE | 93.07±0.17 | 72.96±0.45 |
|  |  | LN | 93.16±0.21 | 72.73±0.25 |
|  |  | SLN | **93.78±0.18** | **73.10±0.46** |
| ResNet-110 | 1.73/1.76M | CE | 93.64±0.16 | 73.90±0.30 |
|  |  | LN | 93.67±0.20 | 73.82±0.22 |
|  |  | SLN | **94.10±0.09** | **73.90±0.27** |

In this search, SLN's CIFAR-100 mean exceeds the stronger endpoint by 0.29,
0.46 and 0.14 percentage points for ResNet-20, -32 and -56, respectively.
ResNet-110 with SLN and CE both round to 73.90%. Reproduce these CIFAR-100
comparisons with [`cifar/reproduce.py`](cifar/reproduce.py); detailed grids
and seed-wise results are in [`cifar/results/`](cifar/results/).

### Comparison with broader methods (CIFAR-100, ResNet-20)

All rows use the same full-training best-test protocol, nine configurations
and five seeds per configuration. Each method uses its selected shared
configuration; the optimizer grid and margin/regularization settings are
listed in [`cifar/README.md`](cifar/README.md).

| Method | Best-test accuracy (%) |
|---|---|
| CE | 69.14±0.22 |
| LN | 69.26±0.21 |
| SLN | 69.55±0.48 |
| NormFace | 69.06±0.12 |
| ArcFace | 68.56±0.17 |
| CosFace | 68.89±0.30 |
| Circle Loss | 66.94±0.14 |
| M-Softmax | 69.64±0.40 |
| L-Softmax | 50.15±0.65 |
| Label smoothing + CE | 69.19±0.46 |
| Label smoothing + SLN | 69.60±0.41 |
| Mixup + CE | 69.69±0.30 |
| Mixup + SLN | 70.44±0.09 |
| SAM + CE | 70.30±0.14 |
| SAM + SLN | **70.53±0.16** |

SAM + SLN reaches 70.53±0.16% and Mixup + SLN reaches 70.44±0.09% in these
configurations. The fixed-optimizer normalized-head comparison is also
included in [`cifar/results/`](cifar/results/).

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
