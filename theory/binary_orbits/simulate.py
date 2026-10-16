"""R08: exact binary logit orbit, independent polar integration and controls.

Float64 CPU only. No manuscript or previous experimental records are changed.
"""
from pathlib import Path
import hashlib
import itertools
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent


def vector(u, gamma, tau, sign=1., norm_epsilon=0.):
    r = np.linalg.norm(u, axis=-1)
    assert np.all(r > 1e-12)
    scale = np.sqrt(2.) / tau / (r + norm_epsilon)**gamma
    wrong = np.exp(-np.logaddexp(0., sign * scale * u[..., 0]))
    e = np.zeros_like(u)
    e[..., 0] = 1.
    radial_filter = gamma * u[..., 0] / (r * (r + norm_epsilon))
    return (sign * scale * wrong)[..., None] * (e - radial_filter[..., None] * u)


def polar(p, gamma, tau, sign=1.):
    r, theta = p[..., 0], p[..., 1]
    a = np.sqrt(2.) / tau
    wrong = np.exp(-np.logaddexp(0., sign * a * r**(1-gamma) * np.cos(theta)))
    return np.stack([sign*a*(1-gamma)*wrong*r**(-gamma)*np.cos(theta),
                     -sign*a*wrong*r**(-gamma-1)*np.sin(theta)], axis=-1)


def log_invariant(u, gamma):
    r = np.linalg.norm(u, axis=-1)
    v = np.abs(u[..., 1])
    # The grid stays off the common-coordinate-zero axis. Endpoint extensions
    # are checked separately, without evaluating 0*log(0).
    assert np.all(v > 0)
    return gamma*np.log(r)+(1-gamma)*np.log(v)


def rk4(fun, x, dt):
    a = fun(x); b = fun(x+dt*a/2); c = fun(x+dt*b/2); d = fun(x+dt*c)
    return x+dt*(a+2*b+2*c+d)/6


def integrate(fun, initial, dt, horizon=12., save_every=.08, euler=False):
    steps = round(horizon/dt); stride = round(save_every/dt)
    assert stride > 0 and abs(steps*dt-horizon) < 1e-10
    assert abs(stride*dt-save_every) < 1e-10
    x = initial.copy(); out = [x.copy()]
    for i in range(steps):
        x = x+dt*fun(x) if euler else rk4(fun, x, dt)
        assert np.isfinite(x).all()
        if (i+1) % stride == 0:out.append(x.copy())
    return np.arange(len(out))*save_every, np.asarray(out)


def main():
    start = time.time()
    cases = [dict(gamma=g,tau=t,radius=r,theta=th,label=lab)
             for g,t,r,th,lab in itertools.product([0.,.2,.4,.7,.9,1.], [.5,1.,2.],
                 [1.,2.], [-2.2,-1.2,-.4,.4,1.2,2.2], [1,2])]
    gamma = np.array([c['gamma'] for c in cases]); tau = np.array([c['tau'] for c in cases])
    signs = np.array([1. if c['label']==1 else -1. for c in cases])
    p0 = np.array([[c['radius'],c['theta']] for c in cases])
    initial = np.stack([p0[:,0]*np.cos(p0[:,1]),p0[:,0]*np.sin(p0[:,1])],axis=-1)
    fun = lambda u:vector(u,gamma,tau,signs)
    values = {}; refinements = []
    for h in [.04,.02,.01,.005]:
        times, xy = integrate(fun,initial,h)
        lc = log_invariant(xy,gamma)
        drift = np.max(np.abs(np.expm1(lc-lc[0])),axis=0)
        refinements.append(dict(step=h,max_relative_invariant_drift=float(drift.max()),
                                median_relative_invariant_drift=float(np.median(drift))))
        values[h] = xy
    xy = values[.005]
    _, pp = integrate(lambda p:polar(p,gamma,tau,signs),p0,.005)
    xy_polar = np.stack([pp[...,0]*np.cos(pp[...,1]),pp[...,0]*np.sin(pp[...,1])],axis=-1)
    polar_error = float(np.max(np.linalg.norm(xy-xy_polar,axis=-1)))
    drift = np.max(np.abs(np.expm1(log_invariant(xy,gamma)-log_invariant(initial,gamma))),axis=0)
    for c,d in zip(cases,drift):c['max_relative_drift'] = float(d)
    # Independent finite differences of the scalar binary loss.
    rng = np.random.default_rng(20261016); fd_error = 0.; directional_error = 0.
    direction_temperature_error = 0.
    for i in range(120):
        r = rng.uniform(.7,3.); th = rng.uniform(-2.8,2.8)
        if abs(np.sin(th)) < .1:th = .5
        point = r*np.array([np.cos(th),np.sin(th)])
        g = float(rng.choice([0.,.2,.4,.7,.9,1.])); t = float(rng.choice([.5,1.,2.])); s = float(rng.choice([-1.,1.]))
        def loss(x):
            margin = s*np.sqrt(2.)*x[0]/t/np.linalg.norm(x)**g
            return np.logaddexp(0.,-margin)
        step=1e-5
        numerical = np.array([(loss(point+step*e)-loss(point-step*e))/(2*step) for e in np.eye(2)])
        vel = vector(point,g,t,s)
        fd_error = max(fd_error,float(np.max(np.abs(numerical+vel))))
        grad_log_c = g*point/r**2+np.array([0.,(1-g)/point[1]])
        directional_error = max(directional_error,float(abs(grad_log_c@vel)/(np.linalg.norm(grad_log_c)*np.linalg.norm(vel))))
        dirs = [vector(point,g,tau0,s) for tau0 in [.5,1.,2.]]
        dirs = [d/np.linalg.norm(d) for d in dirs]
        direction_temperature_error = max(direction_temperature_error,float(max(np.linalg.norm(d-dirs[0]) for d in dirs)))
    # Step refinement for Euler and RK4 at the same clock horizon.
    discretization=[]; single=np.array([.6,1.1])
    for g,method,h in itertools.product([0.,.4,.7,1.],['euler','rk4'],[.16,.08,.04,.02,.01]):
        ts, path = integrate(lambda u:vector(u,g,1.),single,h,horizon=12.8,save_every=.16,euler=method=='euler')
        lc=log_invariant(path,g);err=float(np.max(np.abs(np.expm1(lc-lc[0]))))
        discretization.append(dict(gamma=g,method=method,step=h,max_relative_drift=err))
    # Isotropic / anisotropic constant logit metrics, the implementation's norm
    # epsilon, and L2 regularization. Stress epsilon is labelled explicitly.
    _, base_path=integrate(lambda u:vector(u,.7,1.),single,.005)
    scope={}; scope_paths={}
    matrix=np.array([[2.,.3],[.3,.7]])
    controls={
      'direct':lambda u:vector(u,.7,1.),
      'isotropic_metric':lambda u:2*vector(u,.7,1.),
      'anisotropic_metric':lambda u:matrix@vector(u,.7,1.),
      'epsilon_1e-8':lambda u:vector(u,.7,1.,norm_epsilon=1e-8),
      'epsilon_0.1_stress':lambda u:vector(u,.7,1.,norm_epsilon=.1),
      'l2_0.05':lambda u:vector(u,.7,1.)-.05*u,
    }
    for name,f in controls.items():
        ts,path=integrate(f,single,.005)
        lc=log_invariant(path,.7);err=float(np.max(np.abs(np.expm1(lc-lc[0]))))
        scope[name]=dict(max_relative_invariant_drift=err,final_relative_change=float(np.expm1(lc[-1]-lc[0])))
        if name=='l2_0.05':scope[name]['exponentially_corrected_relative_drift']=float(np.max(np.abs(np.expm1(lc-lc[0]+.05*ts))))
        scope_paths[name]=path
    # Axis cases are separate: on v=0 and u>0, CE/SLN move on the margin
    # axis while LN is stationary; no log invariant with v=0 is evaluated.
    axis_checks=[]
    for g,s,u0 in itertools.product([0.,.4,.7,1.],[-1.,1.],[-2.,2.]):
        vel=vector(np.array([u0,0.]),g,1.,s)
        assert abs(vel[1])<1e-14
        if g==1.:assert np.linalg.norm(vel)<1e-14
        axis_checks.append(dict(gamma=g,label_sign=s,u=u0,velocity=vel.tolist()))
    # Exact finite-step LN norm identity, an algebraic check distinct from
    # a conserved-quantity test of the differential equation.
    u=single;v=vector(u,1.,1.);h=.1
    ln_euler_identity_error=float(abs(np.dot(u+h*v,u+h*v)-np.dot(u,u)-h*h*np.dot(v,v)))
    assert len(cases)==432
    assert fd_error<1e-8 and directional_error<1e-13 and direction_temperature_error<1e-13
    assert drift.max()<2e-8 and polar_error<2e-8
    assert ln_euler_identity_error<1e-14
    for name in ['direct','isotropic_metric']:assert scope[name]['max_relative_invariant_drift']<1e-8
    assert scope['l2_0.05']['exponentially_corrected_relative_drift']<1e-8
    data=dict(times=times,cartesian=xy,polar=xy_polar,initial=initial,
              gammas=gamma,taus=tau,labels=signs,
              **{f'scope_{name}':p for name,p in scope_paths.items()})
    np.savez_compressed(HERE/'trajectories.npz',**data)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary=dict(passed=True,case_count=len(cases),saved_stages=len(times),horizon=12.,step=.005,
                 max_relative_invariant_drift=float(drift.max()),cartesian_polar_max_absolute_difference=polar_error,
                 scalar_loss_gradient_fd_max_error=fd_error,normalized_directional_identity_max_error=directional_error,
                 temperature_unit_direction_max_difference=direction_temperature_error,
                 ln_euler_norm_identity_error=ln_euler_identity_error,
                 refinements=refinements,discretization=discretization,scope=scope,axis_checks=axis_checks,cases=cases,
                 cpu_wall_seconds=time.time()-start,source_sha256=source_sha,
                 trajectory_sha256=hashlib.sha256((HERE/'trajectories.npz').read_bytes()).hexdigest(),
                 seed=20261016,dtype='float64',new_neural_network_training=False)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','discretization','axis_checks']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
