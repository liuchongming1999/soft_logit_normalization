"""Frozen validation-only regularizer grid; detached, resumable single controller."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/data/lcm/rebuttal/jorsc_r02_regularizers_validation_20261018')
PYTHON = '/home/lcm/miniconda3/envs/torch/bin/python'
METHODS = ['ls_ce','ls_sln','mixup_ce','mixup_sln','sam_ce','sam_sln']
PAIRS = [(0.01,0.005),(0.05,0.001),(0.1,0.0001)]
STRENGTHS = {'ls':('ls_eps',[0.05,0.1,0.2]), 'mixup':('mixup_alpha',[0.2,0.4,1.0]), 'sam':('sam_rho',[0.025,0.05,0.1])}
GPUS = list(range(8))
STOP = False

def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')

def write(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)); tmp.replace(path)

def hashes():
    return {name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['original_train.py','train.py','resnet_cifar.py','campaign.py','evaluate_final.py','preflight.py']}

def initialize():
    for d in ['configs','runs','logs','states','failed_attempts','preflight']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():
        m=json.loads((ROOT/'manifest.json').read_text())
        assert m['source_sha256']==hashes(), 'Frozen source changed; inspect before resuming'
        return m
    jobs=[]
    # Balanced launch order across strengths, seeds, and the six families.
    for h in [4,0,1,2,3,5,6,7,8]:
      for seed in range(5):
       for method in METHODS:
        regularizer,base=method.split('_'); key, strengths=STRENGTHS[regularizer]
        lr,wd=PAIRS[h%3]
        jid=f'{method}_h{h:02d}_s{seed}'
        cfg=dict(method=method,seed=seed,setting_id=h,job_id=jid,
                 epochs=160,batch_size=128,workers=4,lr=lr,wd=wd,
                 gamma=0.7 if base=='sln' else 0.0,tau=0.08 if base=='sln' else 1.0,
                 ls_eps=0.0,mixup_alpha=0.0,sam_rho=0.0,
                 data_root='/home/lcm/NeurIPS 2026-Chongming Liu/Code/Soft_logit_normalization/CIFAR10/data',
                 out_dir=str(ROOT/'runs'/jid))
        cfg[key]=strengths[h//3]
        jobs.append(dict(id=jid,config=cfg))
        write(ROOT/'configs'/f'{jid}.json',cfg)
    assert len(jobs)==270
    m=dict(created_at=now(),jobs=jobs,source_sha256=hashes(),gpus=GPUS,
           protocol='45000_train_5000_validation_10000_test_after_selection',
           pipeline_reference='/data/lcm/rebuttal/neurocomputing_revision_c100_20260926_v2',
           data_split='stratified RandomState(12345), sorted indices, 450 train + 50 validation per class',
           data_shuffle='torch.Generator.manual_seed(100000+training_seed)',
           optimizer='SGD momentum 0.9; initial lr /10 at starts of epochs 81 and 121; no warmup',
           precision='bf16 autocast; channels_last; cudnn.benchmark=True',
           selection='earliest highest-validation checkpoint per run; configuration by five-seed mean best validation accuracy',
           ties='validation means within 1e-12 prefer lr=.05/wd=.001, then .01/.005, then smaller setting id',
           evaluation='official test only for locked selected configurations; five seeds; mean and sample SD',
           sam='non-adaptive L2 perturbation; objective gradient excludes weight decay; running BN buffers update only on unperturbed pass',
           completion='270 attempted full runs; failed/nonfinite configurations excluded unless all five seeds finish',
           strengths=STRENGTHS,optimizer_pairs=PAIRS)
    write(ROOT/'manifest.json',m)
    return m

def state(jid):
    p=ROOT/'states'/f'{jid}.json'
    return json.loads(p.read_text()) if p.exists() else dict(status='pending',attempt=0)

def gpu_info():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return {int(r.split(',')[0]):int(r.split(',')[1]) for r in p.stdout.strip().splitlines()}

def halt(*_):
    global STOP
    STOP=True

def run(slots):
    lock=(ROOT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=initialize()
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    # Do not double-launch a worker surviving a prior controller.
    for j in m['jobs']:
        s=state(j['id'])
        if s['status']=='running':
            try: os.kill(s['pid'],0)
            except ProcessLookupError:
                write(ROOT/'states'/f"{j['id']}.json",dict(s,status='interrupted'))
            else: raise RuntimeError(f"Existing worker {s['pid']} for {j['id']}; do not duplicate")
    active={}; started=time.time()
    write(ROOT/'controller.json',dict(pid=os.getpid(),started_at=now(),slots_per_gpu=slots,gpus=GPUS))
    while True:
      for jid,(proc,gpu,job,log,attempt) in list(active.items()):
        code=proc.poll()
        if code is None: continue
        log.close()
        out=Path(job['config']['out_dir'])
        complete=False
        if code==0 and (out/'summary.json').exists():
            import csv
            rows=list(csv.DictReader((out/'history.csv').open()))
            complete=len(rows)==160 and int(rows[-1]['epoch'])==160
        payload=dict(status='complete' if complete else 'failed',pid=proc.pid,gpu=gpu,returncode=code,attempt=attempt,finished_at=now())
        write(ROOT/'states'/f'{jid}.json',payload)
        print(now(),'FINISHED',jid,payload,flush=True)
        del active[jid]
      pending=[]
      for j in m['jobs']:
        s=state(j['id'])
        if s['status'] in ['pending','interrupted'] or (s['status']=='failed' and s.get('attempt',0)<2):
            pending.append((j,s))
      if not STOP and pending:
        free=gpu_info()
        for gpu in GPUS:
          while sum(a[1]==gpu for a in active.values())<slots and pending and free.get(gpu,0)>6000:
            j,s=pending.pop(0); jid=j['id']; out=Path(j['config']['out_dir'])
            attempt=s.get('attempt',0)+1
            if out.exists():
                out.rename(ROOT/'failed_attempts'/f'{jid}_attempt{attempt-1}_{int(time.time())}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            log=(ROOT/'logs'/f'{jid}_attempt{attempt}.log').open('w')
            proc=subprocess.Popen([PYTHON,'-u',str(SOURCE/'train.py'),'--config',str(ROOT/'configs'/f'{jid}.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[jid]=(proc,gpu,j,log,attempt)
            write(ROOT/'states'/f'{jid}.json',dict(status='running',pid=proc.pid,gpu=gpu,attempt=attempt,started_at=now()))
            print(now(),'STARTED',jid,'GPU',gpu,'PID',proc.pid,flush=True)
            # Memory accounting reserves 6 GB before the asynchronous worker allocates.
            free[gpu]-=6000
      counts=Counter(state(j['id'])['status'] for j in m['jobs'])
      progress=dict(updated_at=now(),counts=dict(counts),total=270,elapsed_seconds=time.time()-started,active=[dict(id=jid,pid=a[0].pid,gpu=a[1]) for jid,a in active.items()],stopping=STOP)
      write(ROOT/'progress.json',progress)
      if not active and (STOP or not pending):break
      time.sleep(10)
    if STOP:
        write(ROOT/'controller_stopped.json',progress)
        return
    write(ROOT/'training_finished.json',progress)
    if counts.get('complete',0)!=270:
        raise RuntimeError('Incomplete jobs need inspection before final evaluation')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([PYTHON,'-u',str(SOURCE/'evaluate_final.py')],env=env,check=True)
    write(ROOT/'campaign_complete.json',dict(finished_at=now(),jobs=270,results=str(ROOT/'selected_results.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');parser.add_argument('--slots-per-gpu',type=int,default=2)
    args=parser.parse_args()
    if args.initialize:
        m=initialize();print('INITIALIZED',len(m['jobs']),ROOT)
    else:
        ROOT.mkdir(parents=True,exist_ok=True);run(args.slots_per_gpu)
