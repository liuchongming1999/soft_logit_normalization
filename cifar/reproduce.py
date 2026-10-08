"""Portable scheduler for the frozen nine-configuration CIFAR-100 comparisons."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from protocol import METHODS, make_jobs
from summarize import check_complete, collect

SOURCE = Path(__file__).resolve().parent


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def source_hashes():
    repo = SOURCE.parent
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ['cifar/train_besttest.py', 'cifar/protocol.py',
                         'cifar/models/resnet_cifar.py', 'sln/cosine.py', 'sln/losses.py',
                         'cifar/reproduce.py', 'cifar/summarize.py']}


def initialize(args):
    root = args.out_root.resolve()
    for name in ['runs', 'configs', 'states', 'logs', 'interrupted']:
        (root / name).mkdir(parents=True, exist_ok=True)
    requested = make_jobs(args.depths, args.methods, range(5), args.data_root, root, args.workers)
    if not requested:
        raise ValueError('The selected depths and methods contain no manuscript comparison')
    path = root / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['source_sha256'] != source_hashes():
            raise ValueError('The frozen source changed; choose a new output root')
        existing = {job['id']: job for job in manifest['jobs']}
        for job in requested:
            if job['id'] in existing and existing[job['id']] != job:
                raise ValueError('Frozen configuration differs: ' + job['id'])
            existing[job['id']] = job
        manifest['jobs'] = list(existing.values())
    else:
        manifest = dict(protocol='full50000_best_test', seeds=list(range(5)),
                        configurations_per_group=9, source_sha256=source_hashes(), jobs=requested)
    manifest['jobs_total'] = len(manifest['jobs'])
    atomic(path, manifest)
    for job in requested:
        atomic(root / 'configs' / (job['id'] + '.json'), job)
    return manifest, {job['id'] for job in requested}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('./data'))
    parser.add_argument('--out-root', type=Path, default=Path('./runs/cifar100'))
    parser.add_argument('--depths', type=int, nargs='+', choices=[20, 32, 56, 110], default=[20, 32, 56, 110])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--gpus', default='0', help='comma-separated physical GPU indices')
    parser.add_argument('--jobs-per-gpu', type=int, choices=[1, 2], default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.depths)) != len(args.depths) or len(set(args.methods)) != len(args.methods):
        raise ValueError('Depths and methods must be unique')
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, requested_ids = initialize(args)
    if args.prepare_only:
        print(json.dumps({'manifest': str(root / 'manifest.json'), 'requested_jobs': len(requested_ids),
                          'jobs_total': manifest['jobs_total']}, indent=2))
        return
    gpus = [value.strip() for value in args.gpus.split(',') if value.strip()]
    if not gpus or any(not value.isdigit() for value in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('--gpus must contain distinct nonnegative GPU indices')
    states = {}
    for job in manifest['jobs']:
        state_path = root / 'states' / (job['id'] + '.json')
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='pending', attempts=0)
        if state['status'] == 'complete':
            check_complete(job)
        elif state['status'] == 'running':
            pid = state.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError('A trainer from the previous controller is still active: ' + job['id'])
            state['status'] = 'pending'
        elif state['status'] == 'failed' and job['id'] in requested_ids:
            state['status'] = 'pending'
        states[job['id']] = state
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        print('Finishing active jobs; no further jobs will start.', flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    active = {}
    last_aggregate = 0
    atomic(root / 'controller.json', dict(pid=os.getpid(), gpus=gpus, jobs_per_gpu=args.jobs_per_gpu))
    while True:
        changed = False
        for slot, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            state = states[job['id']]
            try:
                if code != 0:
                    raise ValueError('Training exited with code ' + str(code))
                check_complete(job)
                state.update(status='complete', exit_code=code)
            except Exception as error:
                content = (root / 'logs' / (job['id'] + '.log')).read_text()
                divergence = 'FloatingPointError: Non-finite training/evaluation loss' in content
                state.update(status='diverged' if divergence else 'failed', error=repr(error), exit_code=code)
            atomic(root / 'states' / (job['id'] + '.json'), state)
            print(job['id'] + ': ' + state['status'], flush=True)
            del active[slot]
            changed = True
        if not stopping:
            for gpu in gpus:
                for lane in range(args.jobs_per_gpu):
                    slot = (gpu, lane)
                    if slot in active:
                        continue
                    job = next((job for job in manifest['jobs']
                                if job['id'] in requested_ids and states[job['id']]['status'] == 'pending'), None)
                    if job is None:
                        continue
                    output = Path(job['config']['out_dir'])
                    if output.exists():
                        archive = root / 'interrupted' / (job['id'] + '_' + str(time.time_ns()))
                        shutil.move(str(output), str(archive))
                        old_log = root / 'logs' / (job['id'] + '.log')
                        if old_log.exists():
                            shutil.copy2(old_log, archive / 'train.log')
                    command = [sys.executable, str(SOURCE / 'train_besttest.py')]
                    for key, value in job['config'].items():
                        command.extend(['--' + key.replace('_', '-'), str(value)])
                    log = (root / 'logs' / (job['id'] + '.log')).open('w')
                    environment = os.environ.copy()
                    environment['CUDA_VISIBLE_DEVICES'] = gpu
                    process = subprocess.Popen(command, env=environment, cwd=str(SOURCE), stdout=log, stderr=subprocess.STDOUT)
                    state = states[job['id']]
                    state.update(status='running', attempts=state.get('attempts', 0) + 1,
                                 pid=process.pid, gpu=gpu, lane=lane)
                    atomic(root / 'states' / (job['id'] + '.json'), state)
                    active[slot] = (process, job, log)
                    print(job['id'] + ': started on GPU ' + gpu, flush=True)
                    changed = True
        if changed or time.time() - last_aggregate > 60:
            atomic(root / 'results.json', collect(manifest, states))
            last_aggregate = time.time()
        pending = any(states[job['id']]['status'] == 'pending' for job in manifest['jobs'] if job['id'] in requested_ids)
        if not active and (stopping or not pending):
            break
        time.sleep(2)
    result = collect(manifest, states)
    atomic(root / 'results.json', result)
    print(json.dumps({'counts': result['counts'], 'selected_groups': len(result['finalized'])}, indent=2))
    if any(states[job['id']]['status'] == 'failed' for job in manifest['jobs'] if job['id'] in requested_ids):
        raise SystemExit('Some runs failed; consult their logs and use a new output root or repair the failed run.')


if __name__ == '__main__':
    main()
