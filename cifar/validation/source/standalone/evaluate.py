"""Lock validation-only choices before opening official-test data; cache by hash."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import html
import json
import statistics
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from grid import ROOT, OLD, METHODS, hashes, common_id, tie_rank
from train import evaluate, atomic_json
from reference_train import MEAN, STD
from methods import create_model
from controller import check_complete, state


def select(manifest, depth, panel):
    chosen, grid = {}, {}
    for m in METHODS:
        ids = range(9) if panel == 'tuned' else [common_id(m)]
        settings = []
        for h in ids:
            group = sorted([j for j in manifest['jobs'] if j['depth'] == depth and
                           j['method'] == m and j['setting_id'] == h], key=lambda j: j['seed'])
            assert len(group) == 5 and [j['seed'] for j in group] == list(range(5))
            assert all(state(j)['status'] in {'complete', 'reused'} for j in group)
            summaries = [check_complete(j) for j in group]
            vals = [s['best_val_acc'] for s in summaries]
            settings.append(dict(setting_id=h, config=group[0]['config'],
                                 mean_val=statistics.mean(vals), sd_val=statistics.stdev(vals), jobs=group))
        maximum = max(s['mean_val'] for s in settings)
        tied = [s for s in settings if abs(s['mean_val'] - maximum) < 1e-12]
        chosen[m] = min(tied, key=tie_rank)
        chosen[m]['cooptimal_setting_ids'] = [s['setting_id'] for s in tied]
        grid[m] = [{k: v for k, v in s.items() if k != 'jobs'} for s in settings]
    return dict(depth=depth, panel=panel, selected=chosen, validation_grid=grid,
                selected_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'), source_sha256=hashes(),
                selection_rule=manifest['selection'], test_data_used_for_selection=False)


def old_cache():
    cache = {}
    for name in ['selected_results.json', 'report_panels_20260927.json']:
        path = OLD / name
        if path.exists():
            for panel in json.loads(path.read_text())['panels'].values():
                for item in panel.values():
                    for r in item['per_seed']:
                        key = (r['method'], r['setting_id'], r['seed'])
                        if key in cache:
                            assert cache[key]['test_acc'] == r['test_acc']
                        cache[key] = r
    return cache


def main(depth, panel):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.06)
    target = ROOT / 'evaluation'
    target.mkdir(exist_ok=True)
    with (target / 'evaluator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = target / f'r{depth}_{panel}.json'
        if output.exists():
            print('ALREADY COMPLETE', output); return
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        assert manifest['source_sha256'] == hashes()
        lock_path = target / f'r{depth}_{panel}_selection.json'
        fresh = select(manifest, depth, panel)
        if lock_path.exists():
            selection = json.loads(lock_path.read_text())
            assert selection['selected'] == fresh['selected']
        else:
            selection = fresh
            atomic_json(lock_path, selection)
        # Selection is durably recorded before creating a test dataset.
        cache_dir = target / 'test_cache'; cache_dir.mkdir(exist_ok=True)
        reused_cache = old_cache()
        reuse_audit = {a['id']: a for a in json.loads((ROOT / 'reuse_audit.json').read_text())['runs']}
        loader = None
        rows = []
        for m, setting in selection['selected'].items():
            for j in setting['jobs']:
                cfg = j['config']; cp = Path(cfg['out_dir']) / 'best.pth'
                sha = hashlib.sha256(cp.read_bytes()).hexdigest()
                cached = cache_dir / (j['id'] + '.json')
                if cached.exists():
                    row = json.loads(cached.read_text())
                    assert row['checkpoint_sha256'] == sha
                else:
                    summary = check_complete(j)
                    legacy_key = (m, j['setting_id'], j['seed'])
                    if j['reuse'] and legacy_key in reused_cache:
                        old = reused_cache[legacy_key]
                        assert sha == reuse_audit[j['id']]['checkpoint_sha256']
                        assert old['val_acc'] == summary['best_val_acc']
                        assert old['best_epoch'] == summary['best_epoch']
                        acc, nll = old['test_acc'], old['test_nll']
                        provenance = 'existing official-test evaluation reused'
                    else:
                        if loader is None:
                            tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
                            ds = CIFAR100(cfg['data_root'], train=False, transform=tf, download=False)
                            loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
                        model = create_model(m, depth).cuda().to(memory_format=torch.channels_last)
                        checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
                        assert checkpoint['val_acc'] == summary['best_val_acc']
                        assert checkpoint['epoch'] == summary['best_epoch']
                        model.load_state_dict(checkpoint['state_dict'], strict=True)
                        acc, nll = evaluate(model, loader, cfg)
                        del model, checkpoint
                        provenance = 'official-test evaluation after frozen validation selection'
                    row = dict(job_id=j['id'], method=m, depth=depth, setting_id=j['setting_id'], seed=j['seed'],
                               checkpoint_sha256=sha, best_epoch=summary['best_epoch'], val_acc=summary['best_val_acc'],
                               test_acc=acc, test_nll=nll, provenance=provenance,
                               **{k:cfg[k] for k in ['lr','wd','gamma','tau','margin','scale']})
                    atomic_json(cached, row)
                print('EVALUATED', j['id'], row['test_acc'], row['provenance'], flush=True)
                rows.append(row)
        results = {}
        ce = {r['seed']: r['test_acc'] for r in rows if r['method'] == 'ce'}
        for m in METHODS:
            group = sorted([r for r in rows if r['method'] == m], key=lambda r:r['seed'])
            item = dict(setting_id=group[0]['setting_id'], config={k:group[0][k] for k in ['lr','wd','gamma','tau','margin','scale']}, per_seed=group,
                        cooptimal_setting_ids=selection['selected'][m]['cooptimal_setting_ids'])
            for title, key in [('validation','val_acc'),('test','test_acc')]:
                vals = [r[key] for r in group]
                item[title+'_mean'] = statistics.mean(vals); item[title+'_sd'] = statistics.stdev(vals)
            diffs = [r['test_acc'] - ce[r['seed']] for r in group]
            avg, sd = statistics.mean(diffs), statistics.stdev(diffs)
            half = 2.7764451051977987 * sd / 5**.5
            item['paired_delta_vs_ce'] = dict(mean=avg, sd=sd, ci95=[avg-half,avg+half])
            results[m] = item
        payload = dict(depth=depth, panel=panel, protocol=manifest['protocol'], results=results,
                       completed_at=time.strftime('%Y-%m-%d %H:%M:%S %Z'), selection_file=str(lock_path),
                       mean_sd='sample standard deviation across seeds 0..4')
        with (target / f'r{depth}_{panel}_per_seed.csv').open('w', newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        trs = ''
        for m, r in results.items():
            diff = r['paired_delta_vs_ce']
            trs += f'<tr><td>{m}</td><td>{100*r["validation_mean"]:.3f} ± {100*r["validation_sd"]:.3f}</td><td>{100*r["test_mean"]:.3f} ± {100*r["test_sd"]:.3f}</td><td>{100*diff["mean"]:+.3f}</td><td>{html.escape(json.dumps(r["config"]))}</td></tr>'
        title = f'CIFAR-100 / ResNet-{depth} / {"验证选参" if panel == "tuned" else "同优化器"}'
        page = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>body{{font:16px/1.6 system-ui;padding:36px;color:#173047}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>{title}</h1><p>45k train / 5k validation / 10k official test；5 seeds，均值 ± 样本标准差（%）。全部检查点按验证集选择。差值为相同 seed 下相对 CE 的平均百分点变化。</p><table><tr><th>方法</th><th>验证集</th><th>官方测试集</th><th>相对 CE / pp</th><th>配置</th></tr>{trs}</table><p><a href="r{depth}_{panel}.json">完整统计及配对区间</a> · <a href="r{depth}_{panel}_per_seed.csv">逐 seed 数据</a></p></html>'
        (target / f'r{depth}_{panel}.html').write_text(page)
        # Completion marker last: all tables and records exist at this point.
        atomic_json(output, payload)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--depth', type=int, required=True)
    p.add_argument('--panel', choices=['common','tuned'], required=True)
    a = p.parse_args(); main(a.depth, a.panel)
