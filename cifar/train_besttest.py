"""Audited full-training best-test CIFAR runner.

ResNet-20/32: 160 epochs, no warm-up, LR drops at one-based epochs 81/121.
ResNet-56/110: 200 epochs, five warm-up epochs, LR drops at 101/151.
Use reproduce.py to generate the exact per-depth schedules and nine-setting grids.
The training kernels match those used for the current manuscript tables.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import resnet_cifar as ResNet
from cosine_methods import CosineHead, METHODS as COSINE_METHODS, training_logits


# ---------------------------------------------------------------- losses
def transform_logits(logits, gamma, tau):
    """SLN family: f / (tau * ||f||^gamma). gamma=0,tau=1 recovers CE."""
    if gamma == 0.0 and tau == 1.0:
        return logits
    norms = logits.norm(p=2, dim=1, keepdim=True)
    return logits / (((norms + 1e-8) ** gamma) * tau)


class MSoftmaxLoss(nn.Module):
    """Ensemble soft-margin softmax (Wang et al. 2018): subtract margin m
    from the ground-truth logit."""

    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = margin

    def forward(self, logits, target):
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), target] -= self.margin
        return F.cross_entropy(adjusted, target)


class LSoftmaxLoss(nn.Module):
    """Numerically stable Large-Margin Softmax (Liu et al. 2016)."""

    def __init__(self, in_features, out_features, margin=4,
                 lambda_base=1000.0, lambda_gamma=0.00002, lambda_power=45.0):
        super().__init__()
        self.margin = margin
        self.lambda_base = lambda_base
        self.lambda_gamma = lambda_gamma
        self.lambda_power = lambda_power
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        self.iter = 0
        # T_m coefficients in monomial basis (ascending order)
        coeffs = self._cheb_monomial(margin)
        self.register_buffer('cheb', torch.tensor(coeffs, dtype=torch.float32))

    @staticmethod
    def _cheb_monomial(m):
        if m == 0:
            return [1.0]
        t_prev, t_cur = [1.0], [0.0, 1.0]
        for _ in range(2, m + 1):
            nxt = [0.0] * (len(t_cur) + 1)
            for i, c in enumerate(t_cur):
                nxt[i + 1] += 2.0 * c
            for i, c in enumerate(t_prev):
                nxt[i] -= c
            t_prev, t_cur = t_cur, nxt
        return t_cur

    def cos_m_theta(self, cos_theta):
        # evaluate T_m(cos_theta) via Horner in float32
        x = cos_theta.clamp(-1.0, 1.0)
        out = torch.zeros_like(x)
        for c in reversed(self.cheb.tolist()):
            out = out * x + c
        return out

    def forward(self, features, target):
        # features: (N, in_features); run in fp32 for numerical stability
        with torch.autocast('cuda', enabled=False):
            features = features.float()
            return self._forward_fp32(features, target)

    def _forward_fp32(self, features, target):
        w = self.weight                                 # (C, D); original L-Softmax
        x_norm = features.norm(p=2, dim=1)                # (N,)
        logits_w = F.linear(features, w)                  # |w_j||x| cos(theta_j)
        norm_product = x_norm.unsqueeze(1) * w.norm(p=2, dim=1).unsqueeze(0)
        cos_theta = (logits_w / norm_product.clamp(min=1e-8)).clamp(-1.0, 1.0)

        idx = torch.arange(features.size(0), device=features.device)
        cos_t_y = cos_theta[idx, target]
        # The segment index is a discrete lookup, not a differentiable angle.
        # Backpropagating acos through floor gives 0 * inf at cos(theta)=+/-1.
        # Keep the polynomial/norm gradients while freezing this branch choice.
        theta = torch.acos(cos_t_y.detach())
        k = (self.margin * theta / math.pi).floor().clamp(max=self.margin - 1)
        phi = ((-1.0) ** k) * self.cos_m_theta(cos_t_y) - 2.0 * k
        phi = phi * norm_product[idx, target]

        lam = self.lambda_base / ((1.0 + self.lambda_gamma * self.iter) ** self.lambda_power)
        f_y = (lam * logits_w[idx, target] + phi) / (1.0 + lam)

        logits = logits_w.clone()
        logits[idx, target] = f_y
        self.iter += 1
        return F.cross_entropy(logits, target)


class SAM:
    """Sharpness-aware minimization around an SGD optimizer (Foret et al.)."""

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
        self.param_groups = self.optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self):
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norms.append(torch.norm(p.grad))
        return torch.norm(torch.stack(norms))

    @torch.no_grad()
    def first_step(self):
        norm = self._grad_norm().clamp(min=1e-12)
        scale = self.rho / norm
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                self.optimizer.state[p]['e_w'] = e

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.optimizer.state[p]:
                    continue
                p.sub_(self.optimizer.state[p]['e_w'])
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @property
    def base_optimizer(self):
        return self.optimizer


# ---------------------------------------------------------------- training
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def accuracy(logits, target):
    return (logits.argmax(1) == target).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, tau, gamma, loss_name, max_batches=0):
    model.eval()
    total, correct, nll_sum, brier_sum = 0, 0, 0.0, 0.0
    all_logits, all_labels = [], []
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if loss_name == 'lsoftmax':
                logits = model.raw_logits(x)
            else:
                logits = model(x)
        logits = logits.float()
        tl = transform_logits(logits, gamma, tau) if loss_name in ('ln', 'sln') else logits
        nll_sum += F.cross_entropy(tl, y, reduction='sum').item()
        p = F.softmax(tl, dim=1)
        onehot = F.one_hot(y, logits.size(1)).float()
        brier_sum += ((p - onehot) ** 2).sum(1).sum().item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    return {
        'acc': correct / total,
        'nll': nll_sum / total,
        'brier': brier_sum / total,
        'logits': torch.cat(all_logits),
        'labels': torch.cat(all_labels),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    p.add_argument('--data-root', default='./data')
    p.add_argument('--download', action='store_true', help='download CIFAR when not already present')
    p.add_argument('--arch', default='resnet20')
    p.add_argument('--loss', default='ce',
                   choices=['ce', 'ln', 'sln', 'msoftmax', 'lsoftmax', 'normface', 'arcface', 'cosface', 'circle'])
    p.add_argument('--tau', type=float, default=0.2)
    p.add_argument('--gamma', type=float, default=0.7)
    p.add_argument('--m-margin', type=float, default=0.35)
    p.add_argument('--l-margin', type=int, default=4)
    p.add_argument('--sam-rho', type=float, default=0.0, help='>0 enables SAM')
    p.add_argument('--ls-eps', type=float, default=0.0, help='>0 enables label smoothing')
    p.add_argument('--mixup-alpha', type=float, default=0.0, help='>0 enables mixup')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--milestones', default='81,121', help='one-based LR drop epochs')
    p.add_argument('--warmup-epochs', type=int, default=0)
    p.add_argument('--warmup-start-factor', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val-size', type=int, default=0,
                   help='stratified validation split carved from the train set')
    p.add_argument('--save-ckpt-epochs', default='',
                   help='comma separated epochs at which to save full checkpoints')
    p.add_argument('--save-logits', action='store_true',
                   help='save test/val logits+labels at final and best-val epoch')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--margin', type=float, default=0.)
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--memory-fraction', type=float, default=0.12)
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-test-batches', type=int, default=0)
    args = p.parse_args()
    if args.loss in COSINE_METHODS and (args.sam_rho or args.ls_eps or args.mixup_alpha):
        raise ValueError('This campaign uses standalone cosine-head objectives')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    setup_seed(args.seed)
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True

    num_classes = 10 if args.dataset == 'cifar10' else 100
    mean, std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if args.dataset == 'cifar10' \
        else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds_cls = torchvision.datasets.CIFAR10 if args.dataset == 'cifar10' else torchvision.datasets.CIFAR100

    train_full = ds_cls(root=args.data_root, train=True, transform=train_tf, download=args.download)
    test_set = ds_cls(root=args.data_root, train=False, transform=test_tf, download=args.download)

    targets = np.array(train_full.targets)
    if args.val_size > 0:
        rng = np.random.RandomState(12345)  # fixed split, identical across runs
        val_idx, train_idx = [], []
        for c in range(num_classes):
            idx_c = np.where(targets == c)[0]
            rng.shuffle(idx_c)
            n_val = args.val_size // num_classes
            val_idx.append(idx_c[:n_val])
            train_idx.append(idx_c[n_val:])
        val_idx = np.sort(np.concatenate(val_idx))
        train_idx = np.sort(np.concatenate(train_idx))
        # val loader uses test transform
        val_set_plain = ds_cls(root=args.data_root, train=True, transform=test_tf, download=args.download)
        train_set = torch.utils.data.Subset(train_full, train_idx)
        val_set = torch.utils.data.Subset(val_set_plain, val_idx)
        np.savez(os.path.join(args.out_dir, 'split_indices.npz'),
                 train_idx=train_idx, val_idx=val_idx)
    else:
        train_set = train_full
        val_set = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = (torch.utils.data.DataLoader(
        val_set, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
        if val_set is not None else None)

    # ---- model / loss
    loss_name = args.loss
    backbone = getattr(ResNet, args.arch)(use_sln=False)
    if num_classes != 10:
        backbone.linear = nn.Linear(256, num_classes)
    lsoftmax_loss = None
    if loss_name == 'lsoftmax':
        lsoftmax_loss = LSoftmaxLoss(256, num_classes, margin=args.l_margin).to(device)

        class Wrapped(nn.Module):
            def __init__(self, bb, ls):
                super().__init__()
                self.bb = bb
                self.bb.linear = nn.Identity()
                self.ls = ls  # registered so its weight lands in state_dict

            def forward(self, x):
                # evaluation logits with the original unnormalized weights
                feats = self.bb.feature(x)
                return F.linear(feats, self.ls.weight)

            def feature(self, x):
                return self.bb.feature(x)

            def raw_logits(self, x):
                return self.forward(x)

        model = Wrapped(backbone, lsoftmax_loss)
    elif loss_name in COSINE_METHODS:
        model = CosineHead(backbone, num_classes, args.scale)
    else:
        model = backbone
    model = model.to(device).to(memory_format=torch.channels_last)

    if loss_name == 'msoftmax':
        base_criterion = MSoftmaxLoss(args.m_margin)
    else:
        base_criterion = None

    def criterion(logits, target):
        if loss_name in COSINE_METHODS:
            logits = training_logits(logits / args.scale, target, loss_name, args.margin, args.scale)
        if loss_name in ('ln', 'sln'):
            logits = transform_logits(logits, args.gamma if loss_name == 'sln' else 1.0, args.tau)
        if base_criterion is not None:
            return base_criterion(logits, target)
        return F.cross_entropy(logits, target, label_smoothing=args.ls_eps)

    if loss_name == 'lsoftmax':
        params = model.parameters()  # ls.weight included via Wrapped.ls
    else:
        params = model.parameters()
    if args.sam_rho > 0:
        optimizer = SAM(params, torch.optim.SGD, rho=args.sam_rho,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.wd)

    use_amp = not args.no_amp
    ckpt_epochs = set(int(e) for e in args.save_ckpt_epochs.split(',') if e.strip())
    if 0 in ckpt_epochs:
        torch.save({'epoch': 0, 'state_dict': model.state_dict()},
                   os.path.join(args.out_dir, 'ckpt_ep000.pth'))

    history_path = os.path.join(args.out_dir, 'history.jsonl')
    best_val = -1.0
    best_epoch = -1
    t_start = time.time()

    milestones = [int(e)-1 for e in args.milestones.split(',')]
    for epoch in range(args.epochs):
        if args.warmup_epochs and epoch <= args.warmup_epochs:
            factor = (args.warmup_start_factor + (1-args.warmup_start_factor) *
                      min(epoch/args.warmup_epochs, 1.))
            for g in optimizer.param_groups:
                g['lr'] = args.lr * factor
        if epoch in milestones:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t_ep = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                y_a, y_b = y, y[perm]

            def compute_loss():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    if loss_name == 'lsoftmax':
                        feats = model.feature(x)
                        return lsoftmax_loss(feats, y), None
                    out = model(x)
                    if args.mixup_alpha > 0:
                        return lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b), out
                    return criterion(out, y), out

            if args.sam_rho > 0:
                loss, _ = compute_loss()
                loss.backward()
                optimizer.first_step()
                optimizer.zero_grad()
                # Record BN running moments from the unperturbed pass only.
                bn_momenta = [(module, module.momentum) for module in model.modules()
                              if isinstance(module, nn.modules.batchnorm._BatchNorm)]
                for module, _momentum in bn_momenta:
                    module.momentum = 0.
                try:
                    loss2, _ = compute_loss()
                    loss2.backward()
                finally:
                    for module, momentum in bn_momenta:
                        module.momentum = momentum
                optimizer.second_step()
                optimizer.zero_grad()
                loss = loss.detach()
                out = None
            else:
                loss, out = compute_loss()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            ep_loss += loss.item() * y.size(0)
            if out is not None and args.mixup_alpha == 0:
                ep_correct += (out.detach().argmax(1) == y).sum().item()
            ep_total += y.size(0)

        rec = {'epoch': epoch + 1, 'train_loss': ep_loss / ep_total,
               'train_acc': ep_correct / ep_total if ep_correct else None,
               'lr': optimizer.param_groups[0]['lr'],
               'epoch_time': time.time() - t_ep}

        # evaluation
        test_res = evaluate(model, test_loader, device, args.tau,
                            args.gamma if loss_name == 'sln' else 1.0, loss_name, args.max_test_batches)
        rec['test_acc'] = test_res['acc']
        rec['test_nll'] = test_res['nll']
        if not math.isfinite(rec['train_loss']) or not math.isfinite(rec['test_nll']):
            raise FloatingPointError('Non-finite training/evaluation loss')
        if val_loader is not None:
            val_res = evaluate(model, val_loader, device, args.tau,
                               args.gamma if loss_name == 'sln' else 1.0, loss_name)
            rec['val_acc'] = val_res['acc']
            rec['val_nll'] = val_res['nll']
            select = val_res['acc']
        else:
            val_res = None
            select = test_res['acc']  # current CIFAR-100 protocol: per-seed best-test

        if select > best_val:
            best_val = select
            best_epoch = epoch + 1
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'select_metric': best_val},
                       os.path.join(args.out_dir, 'model_best.pth'))
            if args.save_logits:
                torch.save({'test_logits': test_res['logits'], 'test_labels': test_res['labels'],
                            'val_logits': val_res['logits'] if val_res else None,
                            'val_labels': val_res['labels'] if val_res else None},
                           os.path.join(args.out_dir, 'logits_best.pt'))
        rec['best_epoch'] = best_epoch
        rec['best_select'] = best_val
        with open(history_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        status = dict(rec, pid=os.getpid(), method=loss_name, seed=args.seed)
        temp = os.path.join(args.out_dir, 'status.json.tmp')
        with open(temp, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(temp, os.path.join(args.out_dir, 'status.json'))
        print(json.dumps(status), flush=True)

        if (epoch + 1) in ckpt_epochs:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                       os.path.join(args.out_dir, f'ckpt_ep{epoch+1:03d}.pth'))

    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict()},
               os.path.join(args.out_dir, 'model_final.pth'))
    if args.save_logits:
        final_test = evaluate(model, test_loader, device, args.tau,
                              args.gamma if loss_name == 'sln' else 1.0, loss_name)
        final_val = evaluate(model, val_loader, device, args.tau,
                             args.gamma if loss_name == 'sln' else 1.0, loss_name) if val_loader else None
        torch.save({'test_logits': final_test['logits'], 'test_labels': final_test['labels'],
                    'val_logits': final_val['logits'] if final_val else None,
                    'val_labels': final_val['labels'] if final_val else None},
                   os.path.join(args.out_dir, 'logits_final.pt'))

    summary = {'best_epoch': best_epoch, 'best_select': best_val,
               'final_test_acc': rec['test_acc'],
               'wall_time_sec': time.time() - t_start,
               'protocol': 'train50000_test10000_best_test',
               'train_size': len(train_set), 'test_size': len(test_set),
               'parameters': sum(p.numel() for p in model.parameters()),
               'epochs': args.epochs, 'seed': args.seed, 'method': loss_name,
               'milestones': args.milestones, 'warmup_epochs': args.warmup_epochs,
               'smoke': bool(args.max_train_batches or args.max_test_batches)}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] {args.out_dir} best_epoch={best_epoch} best={best_val:.4f} "
          f"final_test={rec['test_acc']:.4f} time={summary['wall_time_sec']:.0f}s")


if __name__ == '__main__':
    main()
