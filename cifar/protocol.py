"""Frozen CIFAR-100 configuration grids for the current manuscript."""
PAIRS = [(.01, .005), (.05, .001), (.1, .0001)]
CONTROLS = ['ce', 'ln', 'sln', 'normface', 'arcface', 'cosface', 'circle']
EXTRA = ['ls_ce', 'mixup_ce', 'sam_ce', 'ls_sln', 'mixup_sln', 'sam_sln', 'msoftmax', 'lsoftmax']
METHODS = CONTROLS + EXTRA

def settings(method):
    if method in ['ce', 'ln', 'sln']:
        return [dict(lr=lr, wd=wd, gamma=.7 if method=='sln' else (1. if method=='ln' else 0.),
                     tau=.08 if method=='sln' else (.063 if method=='ln' else 1.), margin=0., scale=1.)
                for lr in [.01, .05, .1] for wd in [.0001, .001, .005]]
    if method in CONTROLS:
        values, fixed = {'normface': ([16.,30.,64.], dict(margin=0.)),
                         'arcface': ([.05,.1,.15], dict(scale=16.)),
                         'cosface': ([.1,.2,.3], dict(scale=30.)),
                         'circle': ([.1,.25,.4], dict(scale=64.))}[method]
        return [dict(lr=lr, wd=wd, gamma=0., tau=1., **fixed,
                     **({'scale':v} if method=='normface' else {'margin':v}))
                for v in values for lr,wd in PAIRS]
    if method in ['msoftmax', 'lsoftmax']:
        key = 'm_margin' if method=='msoftmax' else 'l_margin'
        return [dict(lr=lr,wd=wd,gamma=0.,tau=1.,margin=0.,scale=1.,**{key:v})
                for v in ([.5,1.,5.] if method=='msoftmax' else [2,3,4])
                for lr,wd in PAIRS]
    wrapper, loss = method.split('_')
    key, values = {'ls': ('ls_eps', [.05,.1,.2]),
                   'mixup': ('mixup_alpha', [.2,.4,1.]),
                   'sam': ('sam_rho', [.025,.05,.1])}[wrapper]
    return [dict(lr=lr,wd=wd,gamma=.7 if loss=='sln' else 0.,
                 tau=.08 if loss=='sln' else 1.,margin=0.,scale=1.,**{key:v})
            for v in values for lr,wd in PAIRS]


def make_jobs(depths, methods, seeds, data_root, output_root, workers=4):
    from pathlib import Path
    output_root = Path(output_root).resolve()
    data_root = str(Path(data_root).resolve())
    jobs = []
    for depth in depths:
        for method in methods:
            # The manuscript evaluates the broader loss comparisons on ResNet-20.
            if depth != 20 and method not in ['ce', 'ln', 'sln']:
                continue
            for h, hp in enumerate(settings(method)):
                for seed in seeds:
                    job_id = f'r{depth}_{method}_h{h:02d}_s{seed}'
                    loss = method if method in CONTROLS + ['msoftmax', 'lsoftmax'] else method.split('_')[-1]
                    deep = depth >= 56
                    config = dict(hp, dataset='cifar100', arch=f'resnet{depth}', loss=loss,
                                  seed=seed, epochs=200 if deep else 160, batch_size=128,
                                  workers=workers, momentum=.9, val_size=0, data_root=data_root,
                                  out_dir=str(output_root / 'runs' / job_id),
                                  memory_fraction={20: .12, 32: .18, 56: .24, 110: .36}[depth],
                                  milestones='101,151' if deep else '81,121',
                                  warmup_epochs=5 if deep else 0, warmup_start_factor=.1)
                    jobs.append(dict(id=job_id, depth=depth, method=method,
                                     seed=seed, setting_id=h, config=config))
    return jobs
