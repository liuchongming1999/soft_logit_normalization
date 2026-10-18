"""Lock five-seed validation choices before a single evaluation of selected test checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

import train
from campaign import ROOT,METHODS,hashes,write,now

@torch.inference_mode()
def predictions(model,loader,cfg):
    model.eval();ys=[];scores=[]
    for x,y in loader:
        x=x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=train.inference_logits(model,x,cfg)
        ys.append(y.numpy());scores.append(logits.float().cpu().numpy())
    return np.concatenate(ys),np.concatenate(scores)

def main():
    if (ROOT/'selected_results.json').exists():
        raise RuntimeError('Final results already exist; refusing to repeat evaluation')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['source_sha256']==hashes()
    jobs=manifest['jobs'];selection={}
    assert all(json.loads((ROOT/'states'/f"{j['id']}.json").read_text())['status']=='complete' for j in jobs)
    for method in METHODS:
        candidates=[]
        for h in range(9):
            group=sorted([j for j in jobs if j['config']['method']==method and j['config']['setting_id']==h],key=lambda j:j['config']['seed'])
            assert len(group)==5
            vals=[json.loads((Path(j['config']['out_dir'])/'summary.json').read_text())['best_val_acc'] for j in group]
            candidates.append(dict(setting_id=h,mean_val=statistics.mean(vals),sd_val=statistics.stdev(vals),jobs=group))
        highest=max(c['mean_val'] for c in candidates)
        tied=[c for c in candidates if highest-c['mean_val']<=1e-12]
        def tiekey(c):
            cfg=c['jobs'][0]['config'];pair=(cfg['lr'],cfg['wd'])
            return (0 if pair==(.05,.001) else 1 if pair==(.01,.005) else 2,c['setting_id'])
        selection[method]=min(tied,key=tiekey)
    locked=dict(locked_at=now(),selection=selection,all_validation_candidates={m:[dict(setting_id=h,mean_best_val=statistics.mean([json.loads((Path(j['config']['out_dir'])/'summary.json').read_text())['best_val_acc'] for j in jobs if j['config']['method']==m and j['config']['setting_id']==h])) for h in range(9)] for m in METHODS})
    if (ROOT/'selection_locked.json').exists():
        old=json.loads((ROOT/'selection_locked.json').read_text())
        assert old['selection']==selection
    else:write(ROOT/'selection_locked.json',locked)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(train.MEAN,train.STD)])
    ds=CIFAR100(root=jobs[0]['config']['data_root'],train=False,transform=tf,download=False)
    loader=DataLoader(ds,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)
    results={};rows=[]
    receipts=ROOT/'test_receipts';receipts.mkdir(exist_ok=True)
    for method in METHODS:
        per_seed=[]
        for job in selection[method]['jobs']:
            cfg=job['config'];out=Path(cfg['out_dir']);receipt=receipts/f"{job['id']}.json"
            if receipt.exists():
                row=json.loads(receipt.read_text())
                assert row['checkpoint_sha256']==hashlib.sha256((out/'best.pth').read_bytes()).hexdigest()
            else:
                model=train.create_model(method).cuda().to(memory_format=torch.channels_last)
                ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
                model.load_state_dict(ck['state_dict'],strict=True)
                labels,scores=predictions(model,loader,cfg)
                acc=float(np.mean(scores.argmax(1)==labels))
                np.savez_compressed(receipts/f"{job['id']}.npz",labels=labels,scores=scores)
                row=dict(method=method,seed=cfg['seed'],setting_id=cfg['setting_id'],best_epoch=ck['epoch'],val_acc=ck['val_acc'],test_acc=acc,checkpoint_sha256=hashlib.sha256((out/'best.pth').read_bytes()).hexdigest())
                write(receipt,row);del model
                print('TEST',row,flush=True)
            per_seed.append(row);rows.append(row)
        results[method]=dict(config=selection[method]['jobs'][0]['config'],validation_mean=selection[method]['mean_val'],validation_sd=selection[method]['sd_val'],test_mean=statistics.mean(r['test_acc'] for r in per_seed),test_sd=statistics.stdev(r['test_acc'] for r in per_seed),per_seed=per_seed)
    write(ROOT/'selected_results.json',dict(completed_at=now(),protocol=manifest['protocol'],results=results))
    with (ROOT/'selected_per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
