"""Audit and aggregate full-training CIFAR-100 best-test runs."""
import argparse
import json
import math
from pathlib import Path
import statistics


def check_complete(job):
    config = job['config']
    directory = Path(config['out_dir'])
    summary = json.loads((directory / 'summary.json').read_text())
    recorded = json.loads((directory / 'config.json').read_text())
    history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
    if summary['smoke'] or summary['train_size'] != 50000 or summary['test_size'] != 10000:
        raise ValueError('Expected a full-training 50k/10k run')
    if summary['seed'] != job['seed'] or summary['epochs'] != config['epochs']:
        raise ValueError('Seed or epoch count differs from the frozen configuration')
    if [row['epoch'] for row in history] != list(range(1, config['epochs'] + 1)):
        raise ValueError('Epoch history is incomplete')
    for key, value in config.items():
        if key != 'setting_id' and recorded.get(key) != value:
            raise ValueError('Effective argument differs: ' + key)
    for row in history:
        if not all(math.isfinite(row[key]) for key in ['train_loss', 'test_nll', 'test_acc']):
            raise ValueError('Non-finite epoch metric')
        epoch = row['epoch']
        expected_lr = config['lr']
        warmup = config.get('warmup_epochs', 0)
        if warmup and epoch <= warmup + 1:
            start = config.get('warmup_start_factor', .1)
            expected_lr *= start + (1 - start) * min((epoch - 1) / warmup, 1)
        for milestone in map(int, config.get('milestones', '81,121').split(',')):
            if epoch >= milestone:
                expected_lr *= .1
        if not math.isclose(row['lr'], expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Learning-rate history differs at epoch ' + str(epoch))
    best = max(row['test_acc'] for row in history)
    if abs(summary['best_select'] - best) > 1e-14:
        raise ValueError('Summary is not the best-test checkpoint')
    if history[summary['best_epoch'] - 1]['test_acc'] != best:
        raise ValueError('Recorded best epoch differs from history')
    for name in ['model_best.pth', 'model_final.pth']:
        if not (directory / name).is_file():
            raise ValueError('Missing checkpoint: ' + name)
    return summary


def collect(manifest, states):
    completed = {}
    for job in manifest['jobs']:
        if states.get(job['id'], {}).get('status') == 'complete':
            completed[job['id']] = check_complete(job)
    selected, grid, fixed_optimizer = {}, [], {}
    for depth, method in sorted({(job['depth'], job['method']) for job in manifest['jobs']}):
        jobs = [job for job in manifest['jobs'] if (job['depth'], job['method']) == (depth, method)]
        candidates = []
        terminal = all(states.get(job['id'], {}).get('status') in ['complete', 'diverged'] for job in jobs)
        for setting in range(9):
            members = sorted([job for job in jobs if job['setting_id'] == setting], key=lambda job: job['seed'])
            if len(members) != 5 or [job['seed'] for job in members] != list(range(5)):
                raise ValueError('Expected exactly seeds 0-4 for every configuration')
            if not all(job['id'] in completed for job in members):
                continue
            values = [completed[job['id']]['best_select'] for job in members]
            config = {key: value for key, value in members[0]['config'].items()
                      if key not in ['seed', 'out_dir', 'data_root', 'memory_fraction', 'workers']}
            row = dict(group='table6' if depth == 20 else 'table2', depth=depth, method=method,
                       setting_id=setting, settings=config, mean=statistics.mean(values),
                       sd=statistics.stdev(values),
                       seeds=[dict(seed=job['seed'], job_id=job['id'],
                                   accuracy=completed[job['id']]['best_select'],
                                   epoch=completed[job['id']]['best_epoch']) for job in members])
            candidates.append(row)
            grid.append(row)
            fixed_setting = 2 if method in ['ce', 'ln', 'sln'] else 3
            if depth == 20 and method in ['ce', 'ln', 'sln', 'normface', 'arcface', 'cosface', 'circle'] and setting == fixed_setting:
                fixed_optimizer['r20_' + method] = row
        if terminal and candidates:
            winner = dict(max(candidates, key=lambda row: (row['mean'], -row['setting_id'])))
            winner['valid_configurations'] = len(candidates)
            winner['evaluated_configurations'] = 9
            selected[f'r{depth}_{method}'] = winner
    counts = {}
    for job in manifest['jobs']:
        status = states.get(job['id'], {}).get('status', 'pending')
        counts[status] = counts.get(status, 0) + 1
    return dict(protocol='full50000_best_test', accuracy_unit='fraction',
                selection='largest five-seed mean best-test accuracy from nine frozen configurations; smallest configuration index breaks exact ties',
                standard_deviation='sample SD across seeds 0-4',
                finalized=selected, fixed_optimizer=fixed_optimizer, grid_completed=grid,
                completed=len(completed), total=len(manifest['jobs']), counts=counts,
                complete=all(states.get(job['id'], {}).get('status') in ['complete', 'diverged'] for job in manifest['jobs']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    states = {job['id']: json.loads((root / 'states' / (job['id'] + '.json')).read_text())
              for job in manifest['jobs'] if (root / 'states' / (job['id'] + '.json')).exists()}
    result = collect(manifest, states)
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'counts': result['counts'], 'selected_groups': len(result['finalized'])}, indent=2))
    for name, row in sorted(result['finalized'].items()):
        print(f"{name}: {100*row['mean']:.2f} +/- {100*row['sd']:.2f}%, configuration {row['setting_id']}")


if __name__ == '__main__':
    main()
