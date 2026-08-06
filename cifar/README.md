# CIFAR-10 / CIFAR-100

Train ResNet-20/32/56/110 on CIFAR-10/100 with CE / LN / SLN, following the
paper's protocol (He et al. 2016: 160 epochs, batch 128, SGD momentum 0.9,
learning rate ÷10 at epochs 80 and 120, crop+flip augmentation).

## Run

```bash
# SLN on CIFAR-10 (paper setting: gamma=0.7, tau=0.2, lr=0.01, wd=5e-3)
python train.py --dataset cifar10 --arch resnet20 --loss sln \
    --gamma 0.7 --tau 0.2 --lr 0.01 --wd 5e-3 --seed 0

# CE on CIFAR-10 (lr=0.1, wd=1e-4)
python train.py --dataset cifar10 --arch resnet20 --loss ce --lr 0.1 --wd 1e-4

# LN on CIFAR-10 (tau=0.2)
python train.py --dataset cifar10 --arch resnet20 --loss ln --tau 0.2 --lr 0.1 --wd 1e-4

# SLN on CIFAR-100 (tau=0.08)
python train.py --dataset cifar100 --arch resnet20 --loss sln \
    --gamma 0.7 --tau 0.08 --lr 0.01 --wd 5e-3 --seed 0

# LN on CIFAR-100 (tau by the k^(-gamma/2) rule ~= 0.063)
python train.py --dataset cifar100 --arch resnet20 --loss ln --tau 0.063 --lr 0.1 --wd 1e-4

# M-Softmax (margin=0.35) and L-Softmax (margin=4) baselines
python train.py --dataset cifar10 --arch resnet20 --loss msoftmax --m-margin 0.35 --lr 0.1 --wd 1e-4
python train.py --dataset cifar10 --arch resnet20 --loss lsoftmax --l-margin 4 --lr 0.1 --wd 1e-4
```

Datasets download automatically to `--data-root` (default `./data`). Results
(history, summary, final checkpoint) go to `--out-dir` (default `runs/`).

## Expected results (10 runs, mean ± std)

| Model | Loss | CIFAR-10 | CIFAR-100 |
|---|---|---|---|
| ResNet-20 | CE | 91.81±0.15 | 67.26±0.18 |
| | LN | 92.05±0.10 | 67.33±0.16 |
| | L-Softmax | 92.11±0.15 | 67.80±0.16 |
| | M-Softmax | 92.21±0.14 | 67.78±0.15 |
| | SLN | 92.58±0.08 | 69.75±0.25 |
| ResNet-32 | CE | 92.40±0.24 | 68.62±0.33 |
| | LN | 92.45±0.22 | 68.92±0.42 |
| | L-Softmax | 92.78±0.14 | 69.11±0.15 |
| | M-Softmax | 92.76±0.15 | 69.22±0.33 |
| | SLN | 93.19±0.17 | 71.41±0.21 |
| ResNet-56 | CE | 93.07±0.17 | 70.19±0.22 |
| | LN | 93.16±0.21 | 70.32±0.19 |
| | L-Softmax | 93.50±0.14 | 70.82±0.15 |
| | M-Softmax | 93.47±0.14 | 70.71±0.36 |
| | SLN | 93.78±0.18 | 72.76±0.23 |
| ResNet-110 | CE | 93.64±0.16 | 70.84±0.31 |
| | LN | 93.67±0.20 | 71.03±0.20 |
| | L-Softmax | 93.71±0.14 | 71.66±0.15 |
| | M-Softmax | 93.80±0.16 | 71.47±0.52 |
| | SLN | 94.10±0.09 | 73.64±0.29 |

> The paper reports the best over a small learning-rate / weight-decay grid
> (lr ∈ {0.01, 0.05, 0.1}, wd ∈ {1e-4, 1e-3, 5e-3}); the commands above use
> the per-loss optima stated in Appendix B (CE: lr 0.1 / wd 1e-4; SLN: lr 0.01
> / wd 5e-3).

## Files

- `train.py` — unified trainer (argparse, seeds, bf16 autocast, checkpointing)
- `models/resnet_cifar.py` — CIFAR ResNet-20/32/56/110
