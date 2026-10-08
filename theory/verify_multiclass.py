"""Independent double-precision checks of the local multiclass manuscript.

No training accuracy is obtained here. Hessians, moment identities, analytic
trajectories and the normalized maximin criterion are checked on CPU.
"""
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(4)
D=torch.float64


def blocks(k,c):
    one=torch.ones(k,dtype=D)/math.sqrt(k)
    u=torch.zeros(k,dtype=D); u[c]=1.; u-=1/k; u/=u.norm()
    p0=one[:,None]*one[None,:]; pr=u[:,None]*u[None,:]
    return u,pr,torch.eye(k,dtype=D)-pr-p0,p0


def values(k,g,tau,m):
    d=math.sqrt(k/(k-1))*m**(1-g)/tau
    p=1/(1+(k-1)*math.exp(-d)); q=1-p
    a=q*(1-g)*d/m
    lr=q*(1-g)*d/m**2*(g+p*(1-g)*d)
    lt=q/(k-1)/tau**2/m**(2*g)+q*g*d/m**2
    l0=q*g*d/m**2
    return a,(lr,lt,l0),d,p


def loss(z,c,g,tau):
    logits=z/(tau*z.norm(dim=-1,keepdim=True)**g)
    return -torch.log_softmax(logits,dim=-1)[...,c]


def main():
    hchecks=[]; trajectories=[]; moments=[]; balance=[]
    for k in [2,3,4,10,100]:
        for g in [0.,.4,.7,1.]:
            for tau in [.2,1.]:
                m=2.; a,lam,d,p=values(k,g,tau,m)
                for c in sorted({0,k//2,k-1}):
                    u,pr,pt,p0=blocks(k,c); center=m*u
                    h=torch.autograd.functional.hessian(lambda z:loss(z,c,g,tau),center)
                    target=lam[0]*pr+lam[1]*pt+lam[2]*p0
                    v=-torch.func.grad(lambda z:loss(z,c,g,tau))(center)
                    hchecks.append(dict(k=k,gamma=g,tau=tau,c=c,
                                        hessian_abs_error=(h-target).abs().max().item(),
                                        gradient_abs_error=(v-a*u).abs().max().item()))
        if k==2: continue
        for g in [0.,.4,.7,1.]:
            u,pr,pt,p0=blocks(k,0); center=2*u
            a,lam,_,_=values(k,g,1.,2.)
            generator=torch.Generator().manual_seed(10100+k)
            base=torch.randn(1024,k,dtype=D,generator=generator)
            base=base/base.norm(dim=1,keepdim=True)
            base=torch.cat([base,-base])
            h=lam[0]*pr+lam[1]*pt+lam[2]*p0
            for radius in [.01,.03,.1]:
                delta=base*(2*radius); z=(center+delta).requires_grad_(True)
                velocity=-torch.autograd.grad(loss(z,0,g,1.).sum(),z)[0]
                cov=delta.T@delta/len(delta)
                slope=(delta.T@velocity+velocity.T@delta)/len(delta)
                predicted=-h@cov-cov@h
                moments.append(dict(k=k,gamma=g,radius_over_m=radius,
                                    mean_speed=(velocity.mean(0)@u).item(), predicted_speed=a,
                                    mean_error=(velocity.mean(0)-a*u).norm().item(),
                                    covariance_fro_error=(slope-predicted).norm().item(),
                                    covariance_relative_error=((slope-predicted).norm()/predicted.norm().clamp_min(1e-15)).item()))
        for g in [0.,.4,.7]:
            m=torch.tensor(2.,dtype=D,requires_grad=True); tau=1.; m0=1.5
            chi=math.sqrt(k/(k-1)); d=chi*m**(1-g)/tau; d0=chi*m0**(1-g)/tau
            a,lam,_,_=values(k,g,tau,float(m.detach()))
            log_ratio=torch.logaddexp(torch.tensor(d0,dtype=D),torch.tensor(math.log(k-1),dtype=D))-torch.logaddexp(d,torch.tensor(math.log(k-1),dtype=D))
            logs=[2*g*torch.log(m0/m)+2*log_ratio,
                  2*g/(1-g)*torch.log(m0/m)-2*(d-d0)/(k*(1-g)**2),
                  2*g/(1-g)*torch.log(m0/m)]
            derivatives=[torch.autograd.grad(x,m,retain_graph=True)[0].item()*a for x in logs]
            trajectories.append(dict(k=k,gamma=g,log_slope_errors=[abs(x+2*l) for x,l in zip(derivatives,lam)]))
        for eps in [.001,.01,.1]:
            p=math.exp(-eps); d=math.log((k-1)/math.expm1(eps)); b=p*d
            assert b>=.25
            exact=math.sqrt(b)/(1+math.sqrt(b))
            grid=torch.linspace(0,1,100001,dtype=D)
            objective=torch.minimum(grid,(1-grid)*(grid+b*(1-grid)))
            numeric=float(grid[objective.argmax()])
            assert abs(numeric-exact)<=1.1e-5
            balance.append(dict(k=k,epsilon=eps,gamma_exact=exact,gamma_grid=numeric))
    max_h=max(r['hessian_abs_error'] for r in hchecks)
    max_g=max(r['gradient_abs_error'] for r in hchecks)
    max_closed=max(max(r['log_slope_errors']) for r in trajectories)
    assert max_h<1e-11 and max_g<1e-11 and max_closed<1e-11
    # The local covariance error shrinks under a tighter compact cloud.
    for k in [3,4,10,100]:
        for g in [0.,.4,.7,1.]:
            rows=[r for r in moments if r['k']==k and r['gamma']==g]
            assert rows[0]['covariance_fro_error']<rows[-1]['covariance_fro_error']
    result=dict(passed=True,precision='float64',independent_of_network_training=True,
                hessian_cases=len(hchecks),hessian_max_abs_error=max_h,gradient_max_abs_error=max_g,
                analytic_trajectory_max_log_slope_error=max_closed,
                hessian_checks=hchecks,closed_form_checks=trajectories,
                compact_cloud_checks=moments,normalized_balance_checks=balance)
    output=Path(__file__).parent/'multiclass_verification.json'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':main()
