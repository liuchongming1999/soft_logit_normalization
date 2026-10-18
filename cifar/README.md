# CIFAR training and reproducibility

`train_besttest.py` is the audited training implementation for the current
CIFAR-100 manuscript comparisons in Tables 2, 3 and 9. `reproduce.py` creates
the frozen grids, schedules GPU jobs, resumes completed runs, and aggregates
their best-test values. `train.py` remains a generic single-run entry point
for the original CIFAR-10 settings.

## Environment and data

The CIFAR-100 runs used Python with PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128
and CUDA 12.8. The audited trainer requires a CUDA GPU supporting bf16.
Install the appropriate PyTorch/CUDA build for your hardware, then the
remaining repository dependencies. The core SLN package needs only PyTorch.

From the repository root, download CIFAR-100 once:

```bash
python -c "from torchvision.datasets import CIFAR100; CIFAR100('./data', train=True, download=True); CIFAR100('./data', train=False, download=True)"
```

## Reproduce the CIFAR-100 tables

Generate all configurations without starting training:

```bash
python cifar/reproduce.py --data-root ./data --out-root ./runs/cifar100 --prepare-only
```

Run the 24 method/depth groups on your GPUs. There are nine configurations
and five seeds (0-4) per group, giving 1,080 training runs:

```bash
python cifar/reproduce.py --data-root ./data --out-root ./runs/cifar100 --gpus 0,1,2,3,4,5,6,7 --jobs-per-gpu 2
```

Run only a chosen part of the grid:

```bash
python cifar/reproduce.py --data-root ./data --out-root ./runs/cifar100 --depths 20 --methods normface arcface cosface circle --gpus 0
```

Rerunning the same command skips completed runs after checking their configs,
epoch records, summaries and checkpoints. A failed run is retained in an
`interrupted` directory before retry. On an interrupt, active jobs finish and
the scheduler stops launching further jobs. Aggregate existing outputs with:

```bash
python cifar/summarize.py ./runs/cifar100
```

For a single seed at the selected ResNet-20 SLN configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python cifar/train_besttest.py --dataset cifar100 --arch resnet20 --loss sln --gamma 0.7 --tau 0.08 --lr 0.05 --wd 0.001 --workers 4 --epochs 160 --milestones 81,121 --seed 0 --data-root ./data --out-dir ./runs/sln_r20_seed0
```

Each run writes `config.json`, `history.jsonl`, `summary.json`,
`model_best.pth` and `model_final.pth`. Validation size is zero. Test
accuracy is evaluated after every epoch, and the seed's best-test checkpoint
is used. For each method and backbone, select the configuration with the
largest mean over all five seeds; exact ties use the smaller configuration
index. Report the mean and sample standard deviation of the five values.
The included table results are in `results/cifar100_besttest.json`; accuracy
values there are fractions, and the tables below multiply them by 100.

## Shared training settings

All runs use the 50,000-image training pool and the official 10,000-image test
set, batch size 128, SGD momentum 0.9, crop size 32 with padding 4, and a
horizontal-flip probability of 0.5. CIFAR-100 channel means are
(0.5071, 0.4867, 0.4408) and standard deviations are (0.2675, 0.2565, 0.2761).
The incomplete training batch is dropped. Training uses bf16 autocast and
fp32 parameters, channels-last tensors and cudnn benchmarking, with no
gradient clipping. Normalized-head margin calculations are in fp32.

| Depth | Epochs | Warm-up | LR reductions |
|---|---|---|---|
| ResNet-20 / ResNet-32 | 160 | none | after 80 and 120 completed epochs |
| ResNet-56 / ResNet-110 | 200 | five epochs; 0.1x initial LR at epoch 1, full LR at epoch 6 | after 100 and 150 completed epochs |

Each reduction divides the learning rate by ten. `--milestones` accepts
one-based reduction epochs: 81,121 or 101,151.

## Nine configurations per method

- CE, LN and SLN: learning rates {0.01, 0.05, 0.1} crossed with weight decays
  {0.0001, 0.001, 0.005}. LN uses tau=0.063; SLN uses gamma=0.7, tau=0.08.
- Each remaining method: three loss settings crossed with optimizer pairs
  (lr, wd) in {(0.01, 0.005), (0.05, 0.001), (0.1, 0.0001)}.

| Method | Three loss settings | Fixed settings |
|---|---|---|
| NormFace | scales 16, 30, 64 | zero margin |
| ArcFace | angular margins 0.05, 0.1, 0.15 | scale 16, monotonic branch |
| CosFace | cosine margins 0.1, 0.2, 0.3 | scale 30 |
| Circle Loss | margins 0.1, 0.25, 0.4 | scale 64, class-level single-positive form, detached adaptive weights |
| Label smoothing + CE or SLN | epsilon 0.05, 0.1, 0.2 | default SLN settings when used |
| Mixup + CE or SLN | alpha 0.2, 0.4, 1 | default SLN settings when used |
| SAM + CE or SLN | rho 0.025, 0.05, 0.1 | BN running statistics updated on the unperturbed pass only |
| M-Softmax | additive target-logit margins 0.5, 1, 5 | original affine classifier |
| L-Softmax | angular margins 2, 3, 4 | unnormalized bias-free classifier; lambda_t=1000/(1+0.00002*t)^45 |

NormFace, ArcFace, CosFace and Circle predict the largest unmodified cosine
at test time. L-Softmax predicts the largest unnormalized weight-feature
inner product, without a target-dependent angular transformation. The
affine ResNet-20 classifier has 297,524 parameters; the bias-free L-Softmax
and normalized heads have 297,424.

For Table 7's fixed-optimizer panel, use lr=0.01 and wd=0.005, with the
middle loss setting of each normalized-head method. These jobs are already
part of the nine-configuration grids.

## Expected current results

CIFAR-10: ten runs. CIFAR-100: five seeds at the selected shared configuration.

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

### ResNet-20 broader comparisons

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

### ResNet-20 normalized heads and fixed optimizer

| Method | Nine-configuration selection | Fixed lr=0.01, wd=0.005 |
|---|---|---|
| CE | 69.14±0.22 | 68.08±0.36 |
| LN | 69.26±0.21 | 69.26±0.21 |
| SLN | 69.55±0.48 | 69.44±0.32 |
| NormFace | 69.06±0.12 | 69.06±0.12 |
| ArcFace | 68.56±0.17 | 68.00±0.23 |
| CosFace | 68.89±0.30 | 67.57±0.45 |
| Circle Loss | 66.94±0.14 | 60.94±0.27 |

## CIFAR-10 single runs

The original CIFAR-10 settings use 160 epochs and ten seeds, SGD momentum
0.9 and batch size 128. CE uses lr=0.1, wd=0.0001; LN additionally uses
tau=0.2; SLN uses gamma=0.7, tau=0.2, lr=0.01, wd=0.005.

```bash
python cifar/train.py --dataset cifar10 --arch resnet20 --loss sln --gamma 0.7 --tau 0.2 --lr 0.01 --wd 0.005 --seed 0
```

## Implementation verification

```bash
python cifar/verify_implementations.py
```

This CPU check evaluates the SLN endpoints and implementation parity,
L-Softmax's original angular formula and finite-difference gradients,
parallel/antiparallel/zero-feature edge cases, normalized-head logits,
and the SAM perturbation and restoration step.

## Validation-selected comparison

The current Table3 also includes the validation-selected results for all fifteen methods. See [validation/README.md](validation/README.md) for the675-run archive, frozen decisions, original trainers and portable training configuration wrapper. The original multi-depth validation appendix tables have been removed from the article; their research records remain retained.
