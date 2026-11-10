# ImageNet-1K

Train ResNet / ViT on ImageNet-1K with CE / LN / SLN.

`train.py` is the [timm](https://github.com/huggingface/pytorch-image-models)
training script (Ross Wightman) with a small addition: the flags
`--sln --gamma <g> --temperature <t>` apply the SLN transform to the logits
before the loss. Everything else is unchanged timm.

## Training setup

AdamW, base lr 3e-3, weight decay 0.06, batch 1024, 300 epochs, cosine
schedule, 5 warmup epochs, RandAugment `rand-m9-mstd0.5`, random erasing 0.25,
stochastic depth 0.1, mixed precision, no model EMA. Loss settings:
$(\gamma,\tau)=(0,1)$ for CE and $(0.7,0.2)$ for SLN. For LN, use
$(1,0.04)$ with ResNet-50 and $(1,0.2)$ with ResNet-101/152.

## Run

```bash
# CE (baseline)
torchrun --nproc_per_node=4 train.py /path/to/imagenet \
    --model resnet50 -b 256 --sched cosine --epochs 300 --lr 3e-3 \
    --opt adamw --weight-decay 0.06 --warmup-epochs 5 --drop-path 0.1 \
    --aa rand-m9-mstd0.5 --amp --crop-pct 0.875 --reprob 0.25 --resplit --seed 42

# SLN (gamma=0.7, tau=0.2) — add the flags:
#   --sln --gamma 0.7 --temperature 0.2
# LN (gamma=1; tau=0.04 for ResNet-50, 0.2 for ResNet-101/152):
#   --sln --gamma 1.0 --temperature 0.04
```

A ready-made launcher is in [`pretrain.sh`](pretrain.sh). ViT models (e.g.
`vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`) use
the same recipe and flags.

## Results

The ImageNet validation results reported in the doctoral thesis, selected
epochs, saved training histories and a table-recomputation script are in
[`supplementary/thesis/`](../supplementary/thesis/README.md).
