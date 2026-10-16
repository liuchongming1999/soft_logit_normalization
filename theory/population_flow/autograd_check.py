"""Optional independent PyTorch cross-check; CPU only, no training jobs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import dynamics as d


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(20261016)
    errors=[]
    for k in [2,3,10]:
        for gamma in [0.,.4,.7,1.]:
            for tau in [.5,1.]:
                for c in sorted({0,k-1}):
                    z=2*d.axes(k)[0][c]+.2*rng.normal(size=k)
                    x=torch.tensor(z,dtype=torch.float64,requires_grad=True)
                    def loss(v):
                        return -torch.log_softmax(v/(tau*v.norm()**gamma),dim=-1)[c]
                    grad=torch.autograd.grad(loss(x),x)[0].detach().numpy()
                    h=torch.autograd.functional.hessian(loss,x).detach().numpy()
                    errors.append([float(np.max(np.abs(grad+d.field(z,c,gamma,tau)))),
                                   float(np.max(np.abs(h-d.hessian(z,c,gamma,tau))))])
    result=dict(passed=bool(np.max(errors)<1e-12),cases=len(errors),
                gradient_autograd_max=float(np.max(np.array(errors)[:,0])),
                hessian_autograd_max=float(np.max(np.array(errors)[:,1])),
                device='CPU',dtype='float64',
                source_sha256=hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest())
    print(json.dumps(result,indent=2))
    assert result['passed']


if __name__=='__main__':main()
