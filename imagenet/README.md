# ImageNet-1K

Train ResNet / ViT on ImageNet-1K with CE / LN / SLN.

`train.py` is the [timm](https://github.com/huggingface/pytorch-image-models)
training script (Ross Wightman) with a small addition: the flags
`--sln --gamma <g> --temperature <t>` apply the SLN transform to the logits
before the loss. Everything else is unchanged timm.

## Recipe (paper appendix)

AdamW, base lr 3e-3, weight decay 0.06, batch 1024, 300 epochs, cosine
schedule, 5 warmup epochs, RandAugment `rand-m9-mstd0.5`, random erasing 0.25,
stochastic depth 0.1, mixed precision, no model EMA. Loss settings:
$(\gamma,\tau)=(0,1)$ for CE, $(1,0.04)$ for LN, $(0.7,0.2)$ for SLN.

## Run

```bash
# CE (baseline)
torchrun --nproc_per_node=4 train.py /path/to/imagenet \
    --model resnet50 -b 256 --sched cosine --epochs 300 --lr 3e-3 \
    --opt adamw --weight-decay 0.06 --warmup-epochs 5 --drop-path 0.1 \
    --aa rand-m9-mstd0.5 --amp --crop-pct 0.875 --reprob 0.25 --resplit --seed 42

# SLN (gamma=0.7, tau=0.2) — add the flags:
#   --sln --gamma 0.7 --temperature 0.2
# LN (gamma=1, tau=0.04):
#   --sln --gamma 1.0 --temperature 0.04
```

A ready-made launcher is in [`pretrain.sh`](pretrain.sh). ViT models (e.g.
`vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`) use
the same recipe and flags.

## Expected results (single run, validation set)

| Model | #Param | Loss | Top-1 | Top-5 |
|---|---|---|---|---|
| ResNet-50 | 26M | CE | 76.8 | 93.0 |
| | | LN | 78.1 | 93.8 |
| | | SLN | **79.2** | **94.6** |
| ResNet-101 | 45M | CE | 78.6 | 93.9 |
| | | LN | 79.4 | 93.0 |
| | | SLN | **80.5** | **95.3** |
| ResNet-152 | 60M | CE | 78.9 | 94.1 |
| | | LN | 79.9 | 93.2 |
| | | SLN | **81.0** | **95.5** |
