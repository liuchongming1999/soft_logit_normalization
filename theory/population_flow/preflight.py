"""Independent finite-difference checks; optional torch autograd cross-check."""
import json
from pathlib import Path
import numpy as np
from dynamics import axes, population, geometry, field, hessian, rates, closed_prediction


def loss(z, c, g, tau):
    q = z / (tau * np.linalg.norm(z) ** g)
    return np.logaddexp.reduce(q) - q[c]


def main():
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(20261016)
    grad_err, hess_err, spectrum_err, moment_err, distribution_err = [], [], [], [], []
    closed_err = []
    for k in [2,3,10]:
        u, one, pr, pt, p0 = axes(k)
        for gamma in [0.,.4,.7,1.]:
            for tau in [.5,1.]:
                for c in sorted({0,k-1}):
                    z = 2*u[c] + .2*rng.normal(size=k)
                    step = 1e-5
                    grad = np.array([(loss(z+step*e,c,gamma,tau)-loss(z-step*e,c,gamma,tau))/(2*step) for e in np.eye(k)])
                    grad_err.append(float(np.max(np.abs(grad + field(z,c,gamma,tau)))))
                    hnum = np.stack([(-field(z+step*e,c,gamma,tau)+field(z-step*e,c,gamma,tau))/(2*step) for e in np.eye(k)],axis=1)
                    hess_err.append(float(np.max(np.abs(hnum-hessian(z,c,gamma,tau)))))
                    _, lam = rates(k,gamma,tau,2.)
                    expected=lam[0]*pr[c]+lam[1]*pt[c]+lam[2]*p0
                    spectrum_err.append(float(np.max(np.abs(expected-hessian(2*u[c],c,gamma,tau)))))
                z=population(k,.1)
                rec,mu,cov=geometry(z)
                n=z.shape[1]
                # Exact permutation and common-reflection covariance structure.
                blocks=rec[2:5]
                expected=blocks[0]*pr+blocks[2]*p0
                if k>2: expected=expected+blocks[1]/(k-2)*pt
                distribution_err.append(float(np.max(np.abs(cov-expected))))
                distribution_err.append(float(np.max(np.abs(mu-2*u))))
                one_rec,_,_=geometry(z[:1],np.array([0]))
                distribution_err.append(float(np.max(np.abs(rec-one_rec))))
                vel=field(z,np.arange(k)[:,None],gamma,tau)
                d=z-mu[:,None]; dv=vel-vel.mean(1)[:,None]
                slope=(np.einsum('cni,cnj->cij',d,dv)+np.einsum('cni,cnj->cij',dv,d))/n
                eps=1e-5
                _,_,cplus=geometry(z+eps*vel)
                _,_,cminus=geometry(z-eps*vel)
                moment_err.append(float(np.max(np.abs((cplus-cminus)/(2*eps)-slope))))
                _,err=closed_prediction(k,gamma,tau,2.,blocks,np.linspace(0,40,101))
                closed_err.append(err)
    result=dict(passed=True,gradient_fd_max=max(grad_err),hessian_fd_max=max(hess_err),
                center_formula_max=max(spectrum_err),moment_fd_max=max(moment_err),
                exact_symmetry_max=max(distribution_err),closed_vs_ode_max_log_error=max(closed_err),
                dtype='float64',seed=20261016)
    assert result['gradient_fd_max']<1e-8
    assert result['hessian_fd_max']<1e-8
    assert result['center_formula_max']<1e-12
    assert result['moment_fd_max']<1e-9
    assert result['exact_symmetry_max']<1e-12
    assert result['closed_vs_ode_max_log_error']<1e-7
    (root/'preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
