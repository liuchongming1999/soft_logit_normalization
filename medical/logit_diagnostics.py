"""Fixed-sample, pre-normalization binary-logit geometry; population moments use /n."""
import json
from pathlib import Path
import numpy as np


def geometry(raw, labels):
    z = np.asarray(raw, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    assert z.shape == (len(y), 2) and np.isfinite(z).all()
    u = np.column_stack(((z[:, 1]-z[:, 0])/np.sqrt(2), z.sum(1)/np.sqrt(2)))
    groups = []
    for c in [0, 1]:
        a = u[y == c]; mu = a.mean(0); delta = a-mu
        cov = delta.T@delta/len(a); radii = np.linalg.norm(delta, axis=1)
        margin = (1 if c == 1 else -1)*mu[0]
        scale = max(abs(float(mu[0])), 1e-12)
        groups.append(dict(label=c, n=len(a), mean=mu.tolist(), covariance=cov.tolist(),
            variance_margin=float(cov[0, 0]), variance_common=float(cov[1, 1]),
            trace=float(np.trace(cov)), signed_mean_margin=float(margin),
            covariance_correlation=float(cov[0, 1]/max(np.sqrt(cov[0, 0]*cov[1, 1]), 1e-12)),
            concentration=float(np.trace(cov)/scale**2),
            radius_q95_over_boundary_distance=float(np.quantile(radii, .95)/scale),
            radius_max_over_boundary_distance=float(radii.max()/scale),
            error=float(np.mean(z[y == c].argmax(1) != c))))
    means = np.array([g['mean'] for g in groups]); d = means[1]-means[0]
    d2 = max(float(d@d), 1e-24)
    vm = np.mean([g['variance_margin'] for g in groups])
    vc = np.mean([g['variance_common'] for g in groups])
    ncm = np.square(u[:, None, :]-means[None, :, :]).sum(2).argmin(1)
    out = dict(n=len(y), class_counts=[g['n'] for g in groups], classes=groups,
        separation=float(np.sqrt(d2)), squared_separation=d2, margin_mean_gap=float(d[0]),
        common_mean_gap=float(d[1]), mean_symmetry_defect=float(np.linalg.norm(means.sum(0))/np.sqrt(d2)),
        variance_margin=float(vm), variance_common=float(vc),
        variance_margin_over_D2=float(vm/d2), variance_common_over_D2=float(vc/d2),
        cdnv=float((vm+vc)/d2),
        error=float(np.mean(z.argmax(1) != y)), balanced_error=float(np.mean([g['error'] for g in groups])),
        empirical_ncm_error=float(np.mean(ncm != y)),
        mean_raw_norm=float(np.linalg.norm(z, axis=1).mean()))
    # Orthogonal-coordinate invariance and the two-component CDNV decomposition.
    covz = [np.square(z[y == c]-z[y == c].mean(0)).sum()/sum(y == c) for c in [0, 1]]
    dz = z[y == 1].mean(0)-z[y == 0].mean(0)
    direct = sum(covz)/(2*max(float(dz@dz), 1e-24))
    assert np.isclose(out['cdnv'], direct, rtol=1e-10, atol=1e-10)
    return out


def selfcheck():
    z = np.array([[3., 1.], [2., 0.], [1., 2.], [0., 3.], [.5, 2.5]])
    y = np.array([0, 0, 1, 1, 1])
    a = geometry(z, y); b = geometry(3*z, y); c = geometry(z+5, y)
    assert np.isclose(a['cdnv'], b['cdnv'])
    assert a['error'] == c['error']
    assert np.isclose(a['variance_margin'], c['variance_margin'])
    assert np.isclose(a['variance_common'], c['variance_common'])
    assert np.isclose(a['cdnv'], c['cdnv'])
    assert np.isclose(a['cdnv'], a['variance_margin_over_D2']+a['variance_common_over_D2'])
    return True


def record_epoch(model, data, labels, method, out, epoch, predict, state_hash):
    import torch
    selfcheck()
    out = Path(out); (out/'trajectory_predictions').mkdir(exist_ok=True)
    before = state_hash(model.state_dict()) if epoch == 0 else None
    rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
    rows = []
    for split in ['train', 'val']:
        raw, probs = predict(model, data[split], method)
        y = labels[split]
        assert np.array_equal(probs.argmax(1), raw.argmax(1))
        values = geometry(raw, y)
        values.update(epoch=int(epoch), split=split)
        rows.append(values)
        np.savez_compressed(out/'trajectory_predictions'/f'{split}_{epoch:03d}.npz',
            raw_logits=raw, labels=y, indices=np.arange(len(y)))
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    if before is not None: assert before == state_hash(model.state_dict())
    with (out/'trajectory.jsonl').open('a') as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
    return rows
