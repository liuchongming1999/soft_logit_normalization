"""Verify no-regularizer equivalence, shared initialization/splits, and SAM BN semantics."""
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

import train
from campaign import ROOT,write,now,initialize

spec=importlib.util.spec_from_file_location('reference_train',Path(__file__).with_name('original_train.py'))
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def seed(s):
    torch.manual_seed(s);torch.cuda.manual_seed_all(s);np.random.seed(s);random.seed(s)

def main():
    manifest=initialize();data=manifest['jobs'][0]['config']['data_root']
    ds=CIFAR100(root=data,train=True,download=False)
    split=train.train_val_split(ds.targets);rsplit=ref.train_val_split(ds.targets)
    assert all(np.array_equal(a,b) for a,b in zip(split,rsplit))
    assert len(split[0])==45000 and len(split[1])==5000
    assert np.all(np.bincount(np.array(ds.targets)[split[0]]) == 450)
    assert np.all(np.bincount(np.array(ds.targets)[split[1]]) == 50)
    seed(3);base=ref.create_model('ce').cuda().to(memory_format=torch.channels_last)
    seed(3);model=train.create_model('ls_ce').cuda().to(memory_format=torch.channels_last)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert sum(p.numel() for p in model.parameters())==297524
    seed(11);x=torch.randn(16,3,32,32,device='cuda').to(memory_format=torch.channels_last);y=torch.arange(16,device='cuda')%100
    errors={}
    for base_method in ['ce','sln']:
        seed(3);base=ref.create_model(base_method).cuda().to(memory_format=torch.channels_last)
        seed(3);model=train.create_model('ls_'+base_method).cuda().to(memory_format=torch.channels_last)
        a=torch.optim.SGD(base.parameters(),lr=.01,momentum=.9,weight_decay=.005)
        b=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.005)
        cfg=dict(method=base_method,gamma=.7,tau=.08)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(ref.inference_logits(base,x,cfg),y)
        loss.backward();a.step();a.zero_grad(set_to_none=True)
        cfg['method']='ls_'+base_method;cfg['ls_eps']=0
        value,_=train.optimization_step(model,b,x,y,cfg)
        delta=max((p.float()-q.float()).abs().max().item() for p,q in zip(base.state_dict().values(),model.state_dict().values()))
        assert delta<1e-6,(base_method,delta)
        assert abs(value-loss.item())<1e-6
        errors[base_method]=delta
    seed(3);model=train.create_model('sam_sln').cuda().to(memory_format=torch.channels_last)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.SGD(model.parameters(),lr=0,momentum=.9,weight_decay=.005)
    train.optimization_step(model,opt,x,y,dict(method='sam_sln',gamma=.7,tau=.08,sam_rho=.05))
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(m.num_batches_tracked.item()==1 for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    assert all(m.track_running_stats for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
    z=torch.randn(8,100,device='cuda');a=torch.arange(8,device='cuda');b=a.flip(0);lam=.37
    mix=lam*F.cross_entropy(z,a)+(1-lam)*F.cross_entropy(z,b)
    target=lam*F.one_hot(a,100).float()+(1-lam)*F.one_hot(b,100).float()
    assert torch.allclose(mix,F.cross_entropy(z,target))
    smooth=F.cross_entropy(z,a,label_smoothing=.1)
    smooth_manual=.9*F.cross_entropy(z,a)+.1*(-z.log_softmax(1).mean())
    assert torch.allclose(smooth,smooth_manual)
    result=dict(passed_at=now(),no_regularizer_step_max_error=errors,parameter_count=297524,
                exact_split_match=True,exact_initialization_match=True,sam_bn_updates=1,sam_exact_parameter_restore_lr0=True,
                mixup_soft_target_equivalence=True,label_smoothing_formula=True)
    write(ROOT/'preflight'/'unit_checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
