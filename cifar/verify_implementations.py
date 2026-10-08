"""Check inherited controls and the corrected full-norm L-Softmax formula."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import train_besttest as train
from models import resnet_cifar
from cosine_methods import training_logits, CosineHead
from sln import sln_logits, LSoftmaxLoss as PublicLSoftmax


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    checks=[]
    for dtype in [torch.float32,torch.bfloat16]:
        z=torch.randn(5,100,dtype=dtype)
        for gamma,tau in [(0.,1.),(1.,.063),(.7,.08)]:
            torch.testing.assert_close(train.transform_logits(z,gamma,tau),sln_logits(z,gamma,tau),rtol=0,atol=0)
    checks.append('The package and audited trainer use the same CE/LN/SLN map in fp32 and bf16')
    x=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,2,0])
    loss=train.LSoftmaxLoss(5,3,margin=1).double()
    a=loss._forward_fp32(x,y);b=F.cross_entropy(F.linear(x,loss.weight),y)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    checks.append('L-Softmax m=1 equals unnormalized bias-free softmax')
    for m in [2,3,4]:
        criterion=train.LSoftmaxLoss(5,3,margin=m).double();criterion.iter=10000
        raw=F.linear(x,criterion.weight)
        product=x.norm(dim=1)[:,None]*criterion.weight.norm(dim=1)[None,:]
        rows=torch.arange(4);theta=(raw/product).clamp(-1,1).acos()[rows,y]
        k=(m*theta/torch.pi).floor().detach()
        phi=(((-1.)**k)*torch.cos(m*theta)-2*k)*product[rows,y]
        lam=1000/(1+.00002*criterion.iter)**45
        reference=raw.clone();reference[rows,y]=(lam*raw[rows,y]+phi)/(1+lam)
        actual=criterion._forward_fp32(x,y)
        torch.testing.assert_close(actual,F.cross_entropy(reference,y),rtol=1e-9,atol=1e-9)
        gx,gw=torch.autograd.grad(actual,(x,criterion.weight))
        assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
        criterion.iter-=1
        step=1e-5;xp=x.detach().clone();xm=x.detach().clone();xp[0,0]+=step;xm[0,0]-=step
        hi=criterion._forward_fp32(xp,y);criterion.iter-=1
        lo=criterion._forward_fp32(xm,y);criterion.iter-=1
        torch.testing.assert_close((hi-lo)/(2*step),gx[0,0],rtol=1e-4,atol=1e-5)
    checks.append('m=2/3/4 full-norm angular formula and finite-difference gradient')
    for dtype in [torch.float32, torch.float64]:
        for m in [2,3,4]:
            # Exact parallel/antiparallel inputs and zero features expose an
            # acos endpoint singularity hidden by ordinary random-input checks.
            endpoint_x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,0.]],
                                    dtype=dtype,requires_grad=True)
            endpoint_y=torch.tensor([0,0,1,1])
            criterion=train.LSoftmaxLoss(2,2,margin=m).to(dtype);criterion.iter=30000
            with torch.no_grad():criterion.weight.copy_(torch.eye(2,dtype=dtype))
            value=criterion._forward_fp32(endpoint_x,endpoint_y)
            gx,gw=torch.autograd.grad(value,(endpoint_x,criterion.weight))
            assert torch.isfinite(value) and torch.isfinite(gx).all() and torch.isfinite(gw).all(),(dtype,m)
    checks.append('L-Softmax exact angular endpoints and zero-feature gradients are finite in fp32/fp64')
    p=torch.nn.Parameter(torch.tensor([.3,-.5]))
    sam=train.SAM([p],torch.optim.SGD,rho=.05,lr=.01)
    original=p.detach().clone();p.grad=torch.tensor([2.,-1.]);sam.first_step()
    torch.testing.assert_close((p-original).norm(),torch.tensor(.05))
    p.grad=torch.tensor([3.,1.]);sam.second_step()
    torch.testing.assert_close(p,original-.01*torch.tensor([3.,1.]))
    checks.append('SAM restores the unperturbed weights before the SGD update')

    # The reusable package head and the audited training head agree, including gradients.
    for m in [1,2,3,4]:
        reference=train.LSoftmaxLoss(5,3,margin=m).double()
        public=PublicLSoftmax(5,3,margin=m).double()
        public.load_state_dict(reference.state_dict());reference.iter=public.iter=10000
        a=reference._forward_fp32(x,y);b=public._forward_fp32(x,y)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,(x,reference.weight),retain_graph=True)
        gb=torch.autograd.grad(b,(x,public.weight),retain_graph=True)
        for left,right in zip(ga,gb):torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(public.eval_logits(x),F.linear(x,public.weight),rtol=0,atol=0)
    checks.append('Reusable L-Softmax loss, gradients and margin-free inference match the audited trainer')
    cosine=torch.tensor([[.3,-.2,.5],[-.1,.7,.2]],requires_grad=True)
    labels=torch.tensor([0,1]);rows=torch.arange(2)
    for method,margin,scale in [('normface',0.,30.),('arcface',.1,16.),('cosface',.2,30.),('circle',.25,64.)]:
        actual=training_logits(cosine,labels,method,margin,scale)
        reference=cosine.detach().clone()
        if method=='arcface':reference[rows,labels]=torch.cos(torch.acos(reference[rows,labels])+margin)
        elif method=='cosface':reference[rows,labels]-=margin
        elif method=='circle':
            reference=(cosine.detach()+margin).clamp_min(0)*(cosine.detach()-margin)
            positive=cosine.detach()[rows,labels]
            reference[rows,labels]=(1+margin-positive).clamp_min(0)*(positive-(1-margin))
        torch.testing.assert_close(actual,scale*reference,rtol=1e-6,atol=1e-6)
        gradient=torch.autograd.grad(F.cross_entropy(actual,labels),cosine)[0]
        assert torch.isfinite(gradient).all()
    # ArcFace's monotonic branch also has finite gradients near the negative endpoint.
    boundary=torch.tensor([[-.999999,.1,.2]],requires_grad=True)
    value=training_logits(boundary,torch.tensor([0]),'arcface',.15,16.)
    assert torch.isfinite(torch.autograd.grad(value.sum(),boundary)[0]).all()
    checks.append('NormFace, ArcFace, CosFace and class-level Circle match their stated logits and have finite gradients')

    parameters={}
    for d in [20,32,56,110]:
        model=getattr(resnet_cifar,f'resnet{d}')(use_sln=False)
        model.linear=torch.nn.Linear(256,100)
        parameters[str(d)]=sum(p.numel() for p in model.parameters())
    result=dict(passed=True,checks=checks,parameters_cifar100=parameters,
                lsoftmax_author_schedule='1000/(1+0.00002*iteration)^45')
    (Path(__file__).parent/'implementation_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
