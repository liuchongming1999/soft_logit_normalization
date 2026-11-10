# Additional results in the doctoral thesis

The JORSC manuscript reports CIFAR and MedMNIST experiments. This directory
collects the additional ImageNet-1K and GLUE experiments reported in the
doctoral thesis. Training entry points are in [`imagenet/`](../../imagenet/)
and [`glue/`](../../glue/).

## ImageNet-1K

The entries are single runs with seed 42. For each training history, the
checkpoint is selected by the highest validation Top-1, and Top-5 is taken
from the same epoch. CE uses $(\gamma,\tau)=(0,1)$, SLN uses $(0.7,0.2)$,
and LN uses $\gamma=1$ with the temperature shown below. Epoch numbers start
at one. The saved CSV histories use zero-based epoch numbers.

| Model | Loss | Temperature | Selected epoch | Top-1 (%) | Top-5 (%) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | CE | 1 | 298 | 77.44 | 93.28 |
| ResNet-50 | LN | 0.04 | 297 | 78.92 | 94.39 |
| ResNet-50 | SLN | 0.2 | 295 | 79.13 | 94.51 |
| ResNet-101 | CE | 1 | 300 | 78.62 | 93.87 |
| ResNet-101 | LN | 0.2 | 298 | 79.50 | 93.01 |
| ResNet-101 | SLN | 0.2 | 291 | 80.51 | 95.36 |
| ResNet-152 | CE | 1 | 288 | 78.92 | 94.07 |
| ResNet-152 | LN | 0.2 | 294 | 80.03 | 93.13 |
| ResNet-152 | SLN | 0.2 | 291 | 81.03 | 95.40 |

## GLUE

The entries are development-set scores at the selected checkpoint for each
task. QQP and MRPC use $F_1$, CoLA uses Matthews correlation, and the other
tasks use accuracy. The mean is the equally weighted average of eight
unrounded scores, with matched and mismatched MNLI counted separately.

| Model | Loss | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | MRPC | RTE | Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BERT-BASE | CE | 84.48/83.84 | 88.15 | 90.92 | 93.23 | 60.98 | 90.21 | 67.50 | 82.42 |
| BERT-BASE | LN | 85.06/84.33 | 88.37 | 91.82 | 92.78 | 61.33 | 88.29 | 68.78 | 82.59 |
| BERT-BASE | SLN | 84.74/84.88 | 88.49 | 91.73 | 93.46 | 63.95 | 91.40 | 71.87 | 83.81 |
| BERT-LARGE | CE | 86.26/85.58 | 88.64 | 92.40 | 93.46 | 62.35 | 91.26 | 65.36 | 83.16 |
| BERT-LARGE | LN | 86.54/86.13 | 88.84 | 92.57 | 92.78 | 62.70 | 90.17 | 73.93 | 84.21 |
| BERT-LARGE | SLN | 86.77/86.31 | 88.86 | 92.64 | 94.04 | 64.59 | 90.91 | 74.29 | 84.80 |

## Recompute the tables

From the repository root, run:

```bash
python supplementary/thesis/verify_results.py
```

The ImageNet calculation reads the nine saved training histories in
[`imagenet/`](imagenet/). The GLUE calculation reads the selected checkpoint
metrics in [`glue.json`](glue.json).
