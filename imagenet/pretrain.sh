#!/bin/bash

# Set up devices
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
data_dir="/data/lcm/ImageNet"
model_name="resnet18"
seed=42

# Training parameters
batch_size=256
lr_decay_schedule="cosine"
epochs=300
lr=0.003
rand_augment="rand-m9-mstd0.5"
optimizer="adamw"
weight_decay=0.06
warmup_epochs=5
stochastic_depth=0.1
aug_repeats=1
amp="--amp"
crop_pct=0.875
erasing_prob=0.25
resplit="--resplit"
split_bn="--split-bn"

# Optional SLN flag
# use_sln="--sln"
use_sln=""
temperature=0.2
gamma=0.7


torchrun --nproc_per_node=4 train.py ${data_dir} -b ${batch_size} --model ${model_name} \
    --sched ${lr_decay_schedule} --epochs ${epochs} --lr ${lr} --aa ${rand_augment} ${split_bn} \
    --opt ${optimizer} --weight-decay ${weight_decay} --warmup-epochs ${warmup_epochs} --drop-path ${stochastic_depth} \
    --aug-repeats ${aug_repeats} ${amp} --crop-pct ${crop_pct} --reprob ${erasing_prob} ${resplit} \
    ${use_sln} --temperature ${temperature} --gamma ${gamma} --seed ${seed} # --mixup 0.8 --cutmix 1.0 --remode pixel