"""Reproducible CPU-only R04 campaign. Resume by exact config/source hashes."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from dynamics import (axes, field, hessian, rates, population, geometry,
                      closed_prediction, general_moment_prediction, rk4_step, mm, METRICS)

ROOT = Path(__file__).resolve().parent
TIMES = np.linspace(0., 40., 101)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def make_cases():
    cases = []
    for k, g, tau, rho in itertools.product([2, 3, 10], [0., .4, .7, 1.], [.5, 1.], [.01, .03, .1, .3, .6]):
        cases.append(dict(group='baseline', k=k, gamma=g, tau=tau, rho=rho, deformation=0., dt=.05, method='rk4'))
    for k, g, rho, deform in itertools.product([3, 10], [0., .7, 1.], [.1, .3], [.05, .15, .3]):
        cases.append(dict(group='symmetry', k=k, gamma=g, tau=1., rho=rho, deformation=deform, dt=.05, method='rk4'))
    for k, g, tau, rho in itertools.product([2, 3, 10], [0., .4, .7, 1.], [.5, 1.], [.01, .6]):
        cases.append(dict(group='refinement', k=k, gamma=g, tau=tau, rho=rho, deformation=0., dt=.025, method='rk4'))
    for k, g, dt in itertools.product([2, 3, 10], [0., .7, 1.], [.4, .2, .1, .05]):
        cases.append(dict(group='euler', k=k, gamma=g, tau=1., rho=.1, deformation=0., dt=dt, method='euler'))
    for i, c in enumerate(cases):
        c['id'] = f'{i:03d}_{c["group"]}_k{c["k"]}_g{c["gamma"]:g}_tau{c["tau"]:g}_r{c["rho"]:g}_s{c["deformation"]:g}_dt{c["dt"]:g}'
    return cases


def residuals(z, labels, gamma, tau):
    _, mu, cov = geometry(z, labels[:, 0])
    v = field(z, labels, gamma, tau)
    mv = v.mean(axis=1)
    delta, dv = z - mu[:, None], v - mv[:, None]
    exact = (np.einsum('cni,cnj->cij', delta, dv) + np.einsum('cni,cnj->cij', dv, delta)) / z.shape[1]
    h = hessian(mu, labels[:, 0], gamma, tau)
    predicted = -mm(h, cov) - mm(cov, h)
    mean_error = np.linalg.norm(mv - field(mu, labels[:, 0], gamma, tau), axis=1).mean()
    cov_error = np.linalg.norm(exact - predicted, axis=(1,2)).mean()
    relative_cov_error = cov_error / max(np.linalg.norm(exact, axis=(1,2)).mean(), 1e-30)
    return [mean_error, cov_error, relative_cov_error]


def one_case(c):
    k, gamma, tau, dt = c['k'], c['gamma'], c['tau'], c['dt']
    full = population(k, c['rho'], c['deformation'])
    full_initial, mu0, cov0 = geometry(full)
    # Symmetry reduces cost exactly, without a representative-class assumption
    # in the deformed runs, which explicitly transport all K classes.
    z = full if c['deformation'] else full[:1].copy()
    classes = np.arange(k) if c['deformation'] else np.array([0])
    labels = classes[:, None]
    initial = z.copy()
    out, local_residuals = [], []
    max_norm_drift = 0.
    max_common_drift = 0.
    initial_norm = np.linalg.norm(initial, axis=-1)
    common_axis = np.ones(k) / np.sqrt(k)
    initial_common = np.einsum('cni,i->cn', initial, common_axis)
    substeps = int(round((TIMES[1] - TIMES[0]) / dt))
    assert abs(substeps * dt - .4) < 1e-12
    fun = lambda points: field(points, labels, gamma, tau)
    for idx, t in enumerate(TIMES):
        if idx:
            for _ in range(substeps):
                z = rk4_step(fun, z, dt) if c['method'] == 'rk4' else z + dt * fun(z)
        values, _, _ = geometry(z, classes)
        out.append(values)
        local_residuals.append(residuals(z, labels, gamma, tau))
        max_norm_drift = max(max_norm_drift, float(np.max(np.abs(np.linalg.norm(z, axis=-1) - initial_norm))))
        max_common_drift = max(max_common_drift, float(np.max(np.abs(np.einsum('cni,i->cn', z, common_axis) - initial_common))))
    measured = np.asarray(out)
    if k == 2:
        full_initial[3] = 0.
    analytic, closed_error = closed_prediction(k, gamma, tau, full_initial[5], full_initial[2:5], TIMES)
    matrix = general_moment_prediction(mu0, cov0, gamma, tau, TIMES) if c['deformation'] else np.empty((0,6))
    active = [0, 1, 2, 4, 5] if k == 2 else list(range(6))
    initial_scale = analytic[0, active]
    diff = np.abs(measured[:, active] - analytic[:, active])
    normalized = diff / initial_scale[None]
    relative = diff / np.maximum(np.abs(analytic[:, active]), 1e-30)
    row = dict(c)
    row.update(particles_per_class=full.shape[1], represented_classes=len(classes),
               population_class_count=k, observations=len(TIMES),
               closed_ode_max_log_difference=closed_error,
               ln_max_particle_norm_drift=max_norm_drift if gamma == 1 else None,
               ce_max_common_coordinate_drift=max_common_drift if gamma == 0 else None,
               max_cloud_radius_over_center=float(measured[:,6].max()),
               initial_simplex_error=float(measured[0,8]),
               max_simplex_error=float(measured[:,8].max()))
    for j, index in enumerate(active):
        name = METRICS[index]
        row[name + '_sup_initial_normalized_error'] = float(normalized[:,j].max())
        row[name + '_sup_relative_error'] = float(relative[:,j].max())
        row[name + '_final_relative_error'] = float(relative[-1,j])
        if matrix.size:
            row[name + '_matrix_sup_relative_error'] = float(np.max(np.abs(measured[:,index] - matrix[:,index]) / np.maximum(np.abs(matrix[:,index]), 1e-30)))
    return row, dict(time=TIMES, measured=measured, analytic=analytic, matrix_local=matrix,
                     instantaneous_residual=np.asarray(local_residuals), initial_full_population=full,
                     final_transported_population=z, initial_centers=mu0, initial_covariances=cov0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['baseline','symmetry','refinement','euler'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cases = make_cases()
    hashes = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['dynamics.py','run_experiments.py']}
    manifest = dict(version=1, source_sha256=hashes, cases=cases, metric_names=METRICS,
                    times=TIMES.tolist(), dtype='float64', device='CPU',
                    distribution='exact finite compact population; equal weights; shells .4,.7,1; all radial/common signs and contrast vertices',
                    m0=2., flow='dz/dt=-grad_z per-sample CE(z/(tau*norm(z)^gamma))',
                    prediction='initial conditions only; center ODE plus closed variance formulas; no fitted rates/time',
                    initial_normalized_error='max_t |measured-predicted| / predicted(t=0)',
                    relative_error='max_t |measured-predicted| / |predicted(t)|',
                    notes=['K=2 has no contrast block; this block is omitted from errors.',
                           'Symmetry perturbations use one-time anisotropic contrast deformation and RMS radius normalization.',
                           'Symmetry runs also integrate the unrestricted local mean/covariance ODE.',
                           'Class proportions are balanced; no image datasets or neural networks are trained.'])
    manifest_path = ROOT/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['source_sha256'] != hashes:
            raise RuntimeError('Sources changed: use fresh output directory or remove only audited preflight outputs')
    else:
        atomic_json(manifest_path, manifest)
    (ROOT/'results').mkdir(exist_ok=True)
    selected = [c for c in cases if args.only is None or c['group'] == args.only]
    if args.limit:
        selected = selected[:args.limit]
    start = time.time()
    for c in selected:
        record_path = ROOT/'results'/f'{c["id"]}.json'
        if record_path.exists():
            continue
        before = time.time()
        row, data = one_case(c)
        row['wall_seconds'] = time.time() - before
        row['source_sha256'] = hashes
        archive = ROOT/'results'/f'{c["id"]}.npz'
        np.savez_compressed(archive, **data)
        row['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        atomic_json(record_path, row)
        complete = len(list((ROOT/'results').glob('*.json')))
        atomic_json(ROOT/'progress.json', dict(status='running', complete=complete, total=len(cases), last_case=c['id'], pid=os.getpid(), elapsed_seconds=time.time()-start))
        print(f'{complete}/{len(cases)} {c["id"]} {row["wall_seconds"]:.2f}s CDNVsup={row["cdnv_sup_relative_error"]:.4g}', flush=True)
    complete = len(list((ROOT/'results').glob('*.json')))
    atomic_json(ROOT/'progress.json', dict(status='complete' if complete == len(cases) else 'partial', complete=complete, total=len(cases), pid=os.getpid(), elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    main()
