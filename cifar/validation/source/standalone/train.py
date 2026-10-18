"""CIFAR-100: original 45k/5k split; official test is evaluator-only.

Interrupted trials restart from their seed, preserving the interrupted attempt.
This avoids claiming exact resumption of persistent data-loader worker RNGs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from methods import create_model, train_logits, inference_logits, angular_lambda
from reference_train import loaders, train_val_split


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def learning_rate(cfg, epoch, batch, steps):
    factor = .1 ** sum(epoch >= m for m in cfg['milestones'])
    if epoch <= cfg['warmup_epochs']:
        fraction = ((epoch - 1) * steps + batch + 1) / (cfg['warmup_epochs'] * steps)
        factor *= cfg['warmup_start_factor'] + (1 - cfg['warmup_start_factor']) * fraction
    return cfg['lr'] * factor


@torch.inference_mode()
def evaluate(model, loader, cfg, max_batches=0):
    model.eval()
    correct = total = 0
    loss_sum = 0.
    for idx, (x, y) in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        y = y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = inference_logits(model, x, cfg)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite inference logits')
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        total += len(y)
    return correct / total, loss_sum / total


def run(cfg):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    out = Path(cfg['out_dir'])
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'config.json', cfg)
    train_loader, val_loader = loaders(cfg['data_root'], cfg['batch_size'], seed, cfg['workers'])
    train_idx, val_idx = train_val_split(train_loader.dataset.dataset.targets)
    split_hash = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
    atomic_json(out / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), split_sha256=split_hash,
                pid=os.getpid(), started_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                train_size=len(train_idx), validation_size=len(val_idx)))
    model = create_model(cfg['method'], cfg['depth']).cuda().to(memory_format=torch.channels_last)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['wd'])
    history = out / 'history.csv'
    fields = ['epoch', 'lr', 'train_loss', 'train_acc', 'val_acc', 'val_nll', 'epoch_seconds',
              'gradient_norm_max_sampled', 'margin_lambda', 'max_reserved_mib']
    with history.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best_acc, best_epoch, step = -1., 0, 0
    start = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        ep_loss = ep_correct = ep_total = 0
        max_grad = 0.
        ep_start = time.time()
        for batch, (x, y) in enumerate(train_loader):
            if cfg.get('max_train_steps') and batch >= cfg['max_train_steps']:
                break
            lr = learning_rate(cfg, epoch, batch, len(train_loader))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                adjusted, unmodified = train_logits(model, x, y, cfg, step)
                loss = F.cross_entropy(adjusted, y)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f'Nonfinite loss epoch={epoch} step={step}')
            loss.backward()
            if batch % 50 == 0:
                # Monitor; no clipping or method-dependent optimization changes.
                norms = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
                grad = torch.stack(norms).norm().item()
                if not math.isfinite(grad):
                    raise FloatingPointError(f'Nonfinite gradient epoch={epoch} step={step}')
                max_grad = max(max_grad, grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ep_loss += loss_value * len(y)
            ep_correct += (unmodified.argmax(1) == y).sum().item()
            ep_total += len(y)
            step += 1
        val_acc, val_nll = evaluate(model, val_loader, cfg, cfg.get('max_val_batches', 0))
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            atomic_torch(out / 'best.pth', dict(epoch=epoch, state_dict=model.state_dict(), val_acc=val_acc))
        row = dict(epoch=epoch, lr=lr, train_loss=ep_loss / ep_total, train_acc=ep_correct / ep_total,
                   val_acc=val_acc, val_nll=val_nll, epoch_seconds=time.time() - ep_start,
                   gradient_norm_max_sampled=max_grad, margin_lambda=angular_lambda(cfg['method'], step),
                   max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        with history.open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        atomic_json(out / 'status.json', dict(epoch=epoch, best_epoch=best_epoch,
                    best_val_acc=best_acc, last_val_acc=val_acc, method=cfg['method'],
                    depth=cfg['depth'], steps=step, parameters=parameters, **{'last': row}))
        if epoch % 10 == 0 or epoch == cfg['epochs']:
            atomic_torch(out / 'last.pth', dict(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict()))
        print(json.dumps(dict(job_id=cfg['job_id'], **row)), flush=True)
    atomic_json(out / 'summary.json', dict(method=cfg['method'], depth=cfg['depth'], seed=seed,
                best_epoch=best_epoch, best_val_acc=best_acc, final_val_acc=val_acc,
                duration_seconds=time.time() - start, epochs=cfg['epochs'], parameters=parameters,
                split_sha256=split_hash, steps=step, smoke=bool(cfg.get('max_train_steps'))))
    print('COMPLETE', cfg['job_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
