"""Persistent queue, source lock, memory admission, auditable reuse/evaluation."""
from __future__ import annotations
import argparse
import collections
import csv
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from grid import SOURCE, ROOT, OLD, METHODS, OLD_METHODS, DEPTHS, build_jobs, hashes, common_id

GPUS = ['0', '1', '2', '3', '7']
MIN_FREE_MIB = 2400


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def atomic(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def state_path(job):
    return ROOT / 'states' / (job['id'] + '.json')


def state(job):
    p = state_path(job)
    return json.loads(p.read_text()) if p.exists() else {'status': 'pending', 'attempts': 0}


def check_complete(job):
    out = Path(job['config']['out_dir'])
    cfg = job['config']
    summary = json.loads((out / 'summary.json').read_text())
    with (out / 'history.csv').open() as f:
        rows = list(csv.DictReader(f))
    if [int(r['epoch']) for r in rows] != list(range(1, cfg['epochs'] + 1)):
        raise ValueError('Incomplete or noncontinuous history')
    for row in rows:
        if any(not math.isfinite(float(row[k])) for k in ['train_loss', 'train_acc', 'val_acc', 'val_nll']):
            raise ValueError('Nonfinite history')
    best = max(float(r['val_acc']) for r in rows)
    best_epoch = next(int(r['epoch']) for r in rows if float(r['val_acc']) == best)
    if best != summary['best_val_acc'] or best_epoch != summary['best_epoch']:
        raise ValueError('Summary/history mismatch')
    actual = json.loads((out / 'config.json').read_text())
    keys = ['method', 'seed', 'setting_id', 'gamma', 'tau', 'margin', 'scale', 'lr', 'wd',
            'epochs', 'batch_size', 'workers', 'data_root', 'out_dir']
    if not job['reuse']:
        keys = list(cfg)
    if any(actual[k] != cfg[k] for k in keys):
        raise ValueError('Configuration mismatch')
    if not all((out / f).is_file() for f in ['best.pth', 'last.pth']):
        raise ValueError('Missing checkpoint')
    if summary.get('smoke'):
        raise ValueError('Preflight cannot enter main results')
    return summary


def initialize():
    for sub in ['configs', 'runs', 'states', 'logs', 'failed_attempts', 'evaluation']:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    path = ROOT / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if hashes() != manifest['source_sha256']:
            raise RuntimeError('Frozen experiment source changed')
        return manifest
    for evidence in [ROOT / 'unit_checks.json', ROOT / 'preflight' / 'results.json']:
        checked = json.loads(evidence.read_text())
        assert checked['passed']
        for name, sha in checked['source_sha256'].items():
            assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == sha, name
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    old_source = SOURCE.parent / 'cifar100_margin_revision_20260926_v2'
    for name, sha in old_manifest['source_sha256'].items():
        assert hashlib.sha256((old_source / name).read_bytes()).hexdigest() == sha, name
    assert (SOURCE / 'reference_train.py').read_bytes() == (old_source / 'train.py').read_bytes()
    assert (SOURCE / 'resnet_cifar.py').read_bytes() == (old_source / 'resnet_cifar.py').read_bytes()
    jobs = build_jobs()
    audits = []
    for job in jobs:
        atomic(ROOT / 'configs' / (job['id'] + '.json'), job['config'])
        if job['reuse']:
            summary = check_complete(job)
            best = Path(job['config']['out_dir']) / 'best.pth'
            sha = hashlib.sha256(best.read_bytes()).hexdigest()
            audits.append(dict(id=job['id'], original_run=str(best.parent), checkpoint_sha256=sha,
                               epochs=160, best_epoch=summary['best_epoch'], best_val_acc=summary['best_val_acc']))
            atomic(state_path(job), dict(status='reused', verified_at=now(), summary=summary,
                                        checkpoint_sha256=sha, attempts=0))
    manifest = dict(created_at=now(), dataset='CIFAR-100', protocol='train45000_val5000_officialtest_after_selection',
                    methods=METHODS, depths=DEPTHS, seeds=list(range(5)), configurations_per_method=9,
                    source_sha256=hashes(), original_source_sha256=old_manifest['source_sha256'], jobs=jobs,
                    common_optimizer=dict(lr=.01, wd=.005, loss_settings='middle loss setting'),
                    selection='max mean best validation accuracy over seeds 0..4; exact mean ties prefer lr=.05 wd=.001, then lr=.01 wd=.005, then lowest setting ID',
                    old_r20_note='Existing validation and official-test results were known when this extension was designed. The uniform tie rule retains the user-selected cooptimal SLN h04.',
                    execution=dict(gpus=GPUS, minimum_free_mib=MIN_FREE_MIB, per_process_torch_memory_fraction=.06,
                                   retries=2, interrupted_runs='archive attempt and restart from seed',
                                   order='common optimizer first; all remaining configurations next'))
    atomic(ROOT / 'reuse_audit.json', dict(verified_at=now(), count=len(audits), runs=audits))
    atomic(path, manifest)
    return manifest


def alive(pid, token):
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        stat = Path(f'/proc/{pid}/stat').read_text().split()[2]
        return token in cmd and stat != 'Z'
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def gpu_free():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    return {i.strip(): int(f.strip()) for i, f in (line.split(',') for line in output.splitlines())}


def write_progress(manifest):
    rows = []
    totals = collections.Counter()
    counts = {d: collections.Counter() for d in DEPTHS}
    for j in manifest['jobs']:
        s = state(j); totals[s['status']] += 1; counts[j['depth']][s['status']] += 1
        live_path = Path(j['config']['out_dir']) / 'status.json'
        live = json.loads(live_path.read_text()) if live_path.exists() else {}
        rows.append(dict(id=j['id'], depth=j['depth'], method=j['method'], setting_id=j['setting_id'], seed=j['seed'],
                         status=s['status'], epoch=live.get('epoch', ''), best_val=live.get('best_val_acc', ''),
                         gpu=s.get('gpu', ''), attempts=s.get('attempts', 0), error=s.get('error', '')))
    payload = dict(updated_at=now(), totals=dict(totals), per_depth=counts, jobs=rows)
    atomic(ROOT / 'progress.json', payload)
    temp = ROOT / 'jobs.csv.tmp'
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, ROOT / 'jobs.csv')
    tr = ''.join(f'<tr><td>ResNet-{d}</td>' + ''.join(f'<td>{counts[d][s]}</td>' for s in ['reused','complete','running','pending','failed']) + '</tr>' for d in DEPTHS)
    live = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ['id','epoch','gpu','best_val']) + '</tr>' for r in rows if r['status'] == 'running')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CIFAR-100 深度与 margin 对比进度</title>
<style>body{{font:16px/1.65 system-ui;margin:40px auto;max-width:1100px;padding:0 24px;color:#173047;background:#f7f9fc}}table{{border-collapse:collapse;background:white;width:100%;margin:20px 0}}td,th{{padding:10px 14px;border-bottom:1px solid #dbe4ed;text-align:left}}h1{{font-size:28px}}a{{color:#1663a4}}.card{{background:white;padding:20px;border-radius:12px}}</style>
<h1>CIFAR-100：模型深度 × margin 方法</h1><p>服务器快照：{now()} · 共 1,800 次训练记录；225 次既有实验经核对后复用，新增 1,575 次。</p>
<div class="card">45,000 训练 / 5,000 验证 / 10,000 官方测试；每方法每深度 9 配置 × 5 seeds。先跑固定优化器 lr=0.01、wd=0.005，再补齐网格。测试评估按预先锁定的配置执行。</div>
<table><tr><th>模型</th><th>核验复用</th><th>新完成</th><th>运行</th><th>待运行</th><th>失败</th></tr>{tr}</table>
<h2>当前训练</h2><table><tr><th>实验</th><th>Epoch</th><th>GPU</th><th>Best validation（比例）</th></tr>{live}</table>
<p>主表中的最终准确率将在相应面板完成后生成。<a href="progress.json">进度 JSON</a> · <a href="jobs.csv">逐任务状态</a> · <a href="manifest.json">冻结的完整配置</a></p>
<h2>已完成的评估</h2><ul>'''
    for file in sorted((ROOT / 'evaluation').glob('r*_*.html')):
        page += f'<li><a href="evaluation/{file.name}">{file.stem}</a></li>'
    page += '</ul></html>'
    (ROOT / 'progress.html.tmp').write_text(page)
    os.replace(ROOT / 'progress.html.tmp', ROOT / 'progress.html')
    return payload


def ready_evaluations(manifest):
    ready = []
    for d in DEPTHS:
        for panel in ['common', 'tuned']:
            if (ROOT / 'evaluation' / f'r{d}_{panel}.json').exists():
                continue
            group = [j for j in manifest['jobs'] if j['depth'] == d and
                     (panel == 'tuned' or j['setting_id'] == common_id(j['method']))]
            if all(state(j)['status'] in {'complete', 'reused'} for j in group):
                ready.append((d, panel))
    return ready


def run(manifest):
    stopping = False
    def stop(sig, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    atomic(ROOT / 'controller.json', dict(pid=os.getpid(), started_at=now(), source=str(SOURCE), gpus=GPUS))
    children = {}
    evaluation = None
    while True:
        busy = set()
        for j in manifest['jobs']:
            s = state(j)
            if s['status'] != 'running':
                continue
            proc = children.get(j['id'])
            if proc is not None and proc.poll() is not None:
                children.pop(j['id'])
            if alive(s['pid'], j['id'] + '.json'):
                busy.add(s['gpu']); continue
            try:
                summary = check_complete(j)
                atomic(state_path(j), s | dict(status='complete', completed_at=now(), summary=summary))
            except Exception as exc:
                log = ROOT / 'logs' / (j['id'] + '.log')
                tail = log.read_text(errors='replace')[-3000:] if log.exists() else ''
                # Numerical failures are surfaced; only interruptions/resource errors retry.
                retryable = 'FloatingPointError' not in tail and s['attempts'] < 3
                atomic(state_path(j), s | dict(status='pending' if retryable else 'failed', error=str(exc),
                                               log_tail=tail, ended_at=now()))
        if evaluation:
            if evaluation['proc'].poll() is None:
                busy.add(evaluation['gpu'])
            else:
                code = evaluation['proc'].returncode
                atomic(ROOT / 'evaluation' / f"r{evaluation['depth']}_{evaluation['panel']}_execution.json",
                       dict(exit_code=code, finished_at=now(), log=evaluation['log']))
                if code != 0:
                    # Preserve failed evaluation for inspection instead of looping reads.
                    (ROOT / 'evaluation' / f"r{evaluation['depth']}_{evaluation['panel']}.failed").write_text(str(code))
                evaluation = None
        if not stopping:
            free = gpu_free()
            available = [g for g in GPUS if g not in busy and free.get(g, 0) >= MIN_FREE_MIB]
            eval_ready = [(d, p) for d, p in ready_evaluations(manifest)
                          if not (ROOT / 'evaluation' / f'r{d}_{p}.failed').exists()]
            if evaluation is None and available and eval_ready:
                d, panel = eval_ready[0]; gpu = available.pop(0)
                path = ROOT / 'logs' / f'evaluate_r{d}_{panel}.log'
                env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2')
                with path.open('a') as log:
                    proc = subprocess.Popen([sys.executable, '-u', str(SOURCE / 'evaluate.py'), '--depth', str(d), '--panel', panel],
                                             stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                evaluation = dict(proc=proc, gpu=gpu, depth=d, panel=panel, log=str(path))
            pending = [j for j in manifest['jobs'] if state(j)['status'] == 'pending']
            for gpu in available:
                if not pending:
                    break
                j = pending.pop(0); s = state(j)
                run_dir = Path(j['config']['out_dir'])
                if run_dir.exists():
                    archive = ROOT / 'failed_attempts' / f"{j['id']}_attempt{s['attempts']}_{int(time.time())}"
                    os.replace(run_dir, archive)
                cfg_file = ROOT / 'configs' / (j['id'] + '.json')
                env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                with (ROOT / 'logs' / (j['id'] + '.log')).open('a') as log:
                    log.write(f'\nATTEMPT {s["attempts"] + 1} {now()} GPU {gpu}\n'); log.flush()
                    proc = subprocess.Popen([sys.executable, '-u', str(SOURCE / 'train.py'), '--config', str(cfg_file)],
                                             stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                children[j['id']] = proc
                atomic(state_path(j), dict(status='running', pid=proc.pid, gpu=gpu, attempts=s['attempts'] + 1, started_at=now()))
                print('START', j['id'], 'gpu', gpu, 'pid', proc.pid, flush=True)
        progress = write_progress(manifest)
        if stopping and not any(r['status'] == 'running' for r in progress['jobs']) and evaluation is None:
            break
        if not any(r['status'] in {'running', 'pending'} for r in progress['jobs']) and evaluation is None:
            if not ready_evaluations(manifest) or all((ROOT / 'evaluation' / f'r{d}_{p}.failed').exists() for d, p in ready_evaluations(manifest)):
                break
        time.sleep(15)
    atomic(ROOT / 'controller_finished.json', dict(finished_at=now(), stopped=stopping, totals=progress['totals']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--status', action='store_true'); args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'queue.lock').open('w') as lock:
        if args.status:
            print(json.dumps(write_progress(json.loads((ROOT / 'manifest.json').read_text()))['totals'])); sys.exit(0)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = initialize()
        print(json.dumps(write_progress(manifest)['totals']), flush=True)
        if not args.initialize_only:
            run(manifest)
