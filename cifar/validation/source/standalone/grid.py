"""Predeclared equal-size grids. No official-test input is used here."""
from pathlib import Path
import hashlib
import json

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/data/lcm/rebuttal/cifar100_depth_margin_20260927')
OLD = Path('/data/lcm/rebuttal/neurocomputing_revision_c100_20260926_v2')
DATA = '/home/lcm/NeurIPS 2026-Chongming Liu/Code/Soft_logit_normalization/CIFAR10/data'
DEPTHS = [20, 32, 56, 110]
METHODS = ['ce', 'ln', 'sln', 'arcface', 'cosface', 'msoftmax', 'lsoftmax', 'sphereface', 'normface', 'circle']
OLD_METHODS = METHODS[:5]
PAIRS = [(0.01, 0.005), (0.05, 0.001), (0.1, 0.0001)]
SEEDS = list(range(5))


def settings(method):
    base = dict(gamma=0., tau=1., margin=0., scale=1.)
    if method == 'ce':
        return [dict(base, lr=lr, wd=wd) for lr in [.01, .05, .1] for wd in [.0001, .001, .005]]
    loss = {
        'ln': [dict(gamma=1., tau=t) for t in [.04, .063, .1]],
        'sln': [dict(gamma=g, tau=t) for g, t in [(.4, .08), (.7, .08), (.7, .16)]],
        'arcface': [dict(margin=m, scale=16.) for m in [.05, .1, .15]],
        'cosface': [dict(margin=m, scale=30.) for m in [.1, .2, .3]],
        'msoftmax': [dict(margin=m) for m in [.5, 1., 5.]],
        'lsoftmax': [dict(margin=m) for m in [2, 3, 4]],
        'sphereface': [dict(margin=m) for m in [2, 3, 4]],
        'normface': [dict(scale=s) for s in [16., 30., 64.]],
        'circle': [dict(margin=m, scale=64.) for m in [.1, .25, .4]],
    }[method]
    return [dict(base, **(v | dict(lr=lr, wd=wd))) for v in loss for lr, wd in PAIRS]


def common_id(method):
    return 2 if method == 'ce' else 3


def pipeline(depth):
    return dict(epochs=160 if depth <= 32 else 200,
                milestones=[81, 121] if depth <= 32 else [101, 151],
                warmup_epochs=0 if depth <= 32 else 5,
                warmup_start_factor=.1, batch_size=128, workers=4,
                split_seed=12345, train_size=45000, val_size=5000,
                momentum=.9, precision='bf16', augmentation='crop4_flip',
                memory_fraction=.06)


def hashes():
    names = ['grid.py', 'methods.py', 'reference_train.py', 'resnet_cifar.py',
             'train.py', 'controller.py', 'evaluate.py', 'test_methods.py']
    return {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in names}


def build_jobs():
    jobs = []
    for d in DEPTHS:
        for m in METHODS:
            for h, hp in enumerate(settings(m)):
                for s in SEEDS:
                    jid = f'r{d}_{m}_h{h:02d}_s{s}'
                    reuse = d == 20 and m in OLD_METHODS
                    output = OLD / 'runs' / f'{m}_h{h:02d}_s{s}' if reuse else ROOT / 'runs' / jid
                    cfg = dict(hp, **pipeline(d), depth=d, method=m, seed=s, setting_id=h,
                               data_root=DATA, out_dir=str(output), job_id=jid)
                    jobs.append(dict(id=jid, depth=d, method=m, setting_id=h, seed=s,
                                     reuse=reuse, config=cfg))
    # First: complete the common-optimizer panel across depths and methods.
    # Next: fill all other configurations, with seed 0 leading each round.
    priority_methods = ['sln', 'ce', 'ln', *METHODS[3:]]
    jobs.sort(key=lambda j: (j['setting_id'] != common_id(j['method']), j['seed'],
                            priority_methods.index(j['method']), [32, 56, 110, 20].index(j['depth']), j['setting_id']))
    assert len(jobs) == 1800 and sum(j['reuse'] for j in jobs) == 225
    return jobs


def tie_rank(setting):
    # Uniform, frozen tie-break: prefer (.05,.001), then (.01,.005), then ID.
    c = setting['config']
    pair = (c['lr'], c['wd'])
    return (0 if pair == (.05, .001) else 1 if pair == (.01, .005) else 2, setting['setting_id'])
