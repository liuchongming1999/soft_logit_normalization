# ImageNet-1K

Train ResNet / ViT on ImageNet-1K with CE / LN / SLN.

`train.py` is the [timm](https://github.com/huggingface/pytorch-image-models)
training script (Ross Wightman) with a small addition: the flags
`--sln --gamma <g> --temperature <t>` apply the SLN transform to the logits
before the loss. Everything else is unchanged timm.

## Recipe (paper, Appendix B)

AdamW, base lr 3e-3, weight decay 0.06, batch 1024, 300 epochs, cosine
schedule, 5 warmup epochs, RandAugment `rand-m9-mstd0.5`, random erasing 0.25,
stochastic depth 0.1, mixed precision.

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

## Expected results (top-1 %)

| Model | #Param | CE | LN | SLN |
|---|---|---|---|---|
| ResNet-50 | 26M | 76.8 | 78.1 | **79.2** |
| ResNet-101 | 45M | 78.6 | 79.4 | **80.5** |
| ResNet-152 | 60M | 78.9 | 79.9 | **81.0** |

> Note: the recipe table in the paper lists EMA, but the released runs were
> trained without it; the commands above match the released runs.
