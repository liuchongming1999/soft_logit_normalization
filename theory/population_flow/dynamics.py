"""R04: independent-logit flow and manuscript moment predictions, float64.

The compact population is a finite probability distribution with exact symmetry,
not a Monte Carlo approximation to a Gaussian. Time is per-particle gradient
flow time, without division by sample count. No rates or time scales are fitted.
"""
import itertools
import math
import numpy as np


def mm(a, b):
    # Explicit contractions avoid macOS Accelerate's spurious floating-point
    # flags on very small matrices; no BLAS threading is needed here.
    return np.einsum('...ij,...jk->...ik', a, b)


def axes(k):
    common = np.ones(k) / np.sqrt(k)
    u = np.sqrt(k / (k - 1)) * (np.eye(k) - np.ones((k, k)) / k)
    pr = u[:, :, None] * u[:, None, :]
    p0 = np.outer(common, common)
    pt = np.eye(k)[None] - pr - p0[None]
    return u, common, pr, pt, p0


def field(z, labels, gamma, tau):
    r2 = np.sum(z * z, axis=-1, keepdims=True)
    if np.min(r2) < 1e-20:
        raise FloatingPointError('Particle approached the logit origin')
    alpha = r2 ** (-gamma / 2) / tau
    logits = alpha * z
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    residual = exp / exp.sum(axis=-1, keepdims=True)
    residual = residual - np.eye(z.shape[-1])[labels]
    return -alpha * (residual - gamma * z * np.sum(z * residual, axis=-1, keepdims=True) / r2)


def hessian(z, labels, gamma, tau):
    """Exact complete-loss Hessian, including normalization second derivative."""
    k = z.shape[-1]
    r2 = np.sum(z * z, axis=-1)
    alpha = r2 ** (-gamma / 2) / tau
    logits = alpha[..., None] * z
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = exp / exp.sum(axis=-1, keepdims=True)
    q = p - np.eye(k)[labels]
    zz = z[..., :, None] * z[..., None, :]
    eye = np.eye(k)
    jac = alpha[..., None, None] * (eye - gamma * zz / r2[..., None, None])
    soft = eye * p[..., :, None] - p[..., :, None] * p[..., None, :]
    dot = np.sum(q * z, axis=-1)
    second = -gamma * alpha[..., None, None] / r2[..., None, None] * (
        q[..., :, None] * z[..., None, :] + z[..., :, None] * q[..., None, :]
        + dot[..., None, None] * eye
        - (gamma + 2) * dot[..., None, None] * zz / r2[..., None, None])
    return mm(mm(jac, soft), jac) + second


def rates(k, gamma, tau, m):
    d = np.sqrt(k / (k - 1)) * m ** (1 - gamma) / tau
    wrong = (k - 1) * np.exp(-d) / (1 + (k - 1) * np.exp(-d))
    p = 1 - wrong
    a = wrong * (1 - gamma) * d / m
    common = wrong * gamma * d / m ** 2
    radial = a / m * (gamma + p * (1 - gamma) * d)
    tangent = wrong / (k - 1) / tau ** 2 / m ** (2 * gamma) + common
    return a, np.array([radial, tangent, common])


def population(k, rho, deformation=0., m0=2.):
    """Product of radial/common signs and wrong-class contrast vertices.

    Every shell has bounded radius rho*m0*shell. Covariance has equal total
    trace in the 2 (binary) or 3 (multiclass) blocks. Other-class permutations
    and reflection in the common coordinate are exact finite-set symmetries.
    """
    u, common, _, _, _ = axes(k)
    centers = m0 * u.copy()
    if deformation:
        # An anisotropic contrast-space map changes relative angles and radii.
        # Recenter and rescale once so RMS center radius stays m0 at t=0.
        transform = np.eye(k) + deformation * np.outer(u[0], u[0])
        centers = mm(centers, transform)
        centers -= centers.mean(axis=0)
        centers *= m0 / np.sqrt(np.mean(np.sum(centers ** 2, axis=1)))
    clouds = []
    for c in range(k):
        vertices = []
        if k > 2:
            others = [j for j in range(k) if j != c]
            mean_other = np.eye(k)[others].mean(axis=0)
            for j in others:
                v = np.eye(k)[j] - mean_other
                v /= np.linalg.norm(v)
                vertices.extend([v, -v])
        else:
            vertices = [np.zeros(k)]
        points = []
        for shell, sr, s0, tangent in itertools.product(
                [.4, .7, 1.], [-1., 1.], [-1., 1.], vertices):
            delta = (sr * u[c] + s0 * common + tangent) / np.sqrt(3 if k > 2 else 2)
            points.append(centers[c] + rho * m0 * shell * delta)
        clouds.append(points)
    return np.asarray(clouds, dtype=np.float64)


def geometry(z, classes=None):
    """Population moments, equal class weights, normalization 1/N_c."""
    k = z.shape[-1]
    if classes is None:
        classes = np.arange(k)
    u, common, _, _, _ = axes(k)
    mu = z.mean(axis=1)
    delta = z - mu[:, None, :]
    cov = np.einsum('cni,cnj->cij', delta, delta) / z.shape[1]
    radial = np.einsum('cni,ci->cn', delta, u[classes])
    common_coord = np.einsum('cni,i->cn', delta, common)
    tangent = delta - radial[..., None] * u[classes, None, :] - common_coord[..., None] * common
    blocks = np.stack([np.mean(radial ** 2, axis=1),
                       np.mean(np.sum(tangent ** 2, axis=-1), axis=1),
                       np.mean(common_coord ** 2, axis=1)], axis=-1)
    radius = np.sqrt(np.sum((z - mu[:, None]) ** 2, axis=-1)).max(axis=1)
    if len(classes) == 1:
        # Symmetry completes the representative class by logit permutations.
        full_mu = np.stack([mu[0][np.array([c if j == 0 else 0 if j == c else j for j in range(k)])]
                            for c in range(k)])
        full_blocks = np.repeat(blocks, k, axis=0)
    else:
        full_mu, full_blocks = mu, blocks
    i, j = np.triu_indices(k, 1)
    distances2 = np.sum((full_mu[i] - full_mu[j]) ** 2, axis=1)
    trace = full_blocks.sum(axis=1)
    cdnv = np.mean((trace[i] + trace[j]) / (2 * distances2))
    distance = np.sqrt(distances2.mean())
    global_mean = full_mu.mean(axis=0)
    rms_radius = np.sqrt(np.mean(np.sum((full_mu - global_mean) ** 2, axis=1)))
    alignment = np.einsum('ci,ci->c', mu, u[classes]) / np.linalg.norm(mu, axis=1)
    gram = np.einsum('ci,di->cd', full_mu - global_mean, full_mu - global_mean) / rms_radius ** 2
    ideal_gram = np.einsum('ci,di->cd', u, u)
    record = np.r_[distance, cdnv, blocks.mean(axis=0), rms_radius,
                   np.max(radius / np.linalg.norm(mu, axis=1)),
                   np.mean(alignment), np.linalg.norm(gram - ideal_gram) / np.linalg.norm(ideal_gram)]
    return record, mu, cov


METRICS = ['separation_rms', 'cdnv', 'variance_radial', 'variance_contrast',
           'variance_common', 'center_rms_radius', 'max_cloud_radius_over_center',
           'mean_radial_alignment', 'simplex_gram_relative_error']


def rk4_step(fun, x, step):
    a = fun(x)
    b = fun(x + step * a / 2)
    c = fun(x + step * b / 2)
    d = fun(x + step * c)
    return x + step / 6 * (a + 2 * b + 2 * c + d)


def closed_prediction(k, gamma, tau, m0, initial_blocks, times, step=.025):
    """Integrate only center ODE; evaluate closed variance-vs-radius formula."""
    m = float(m0)
    log_ratio = np.zeros(3)
    d0 = np.sqrt(k / (k - 1)) * m0 ** (1 - gamma) / tau
    out, checks = [], []
    interval = times[1] - times[0]
    substeps = int(round(interval / step))
    assert abs(substeps * step - interval) < 1e-12
    for idx, t in enumerate(times):
        if idx:
            def ode(x):
                a, lam = rates(k, gamma, tau, x[0])
                return np.r_[a, -2 * lam]
            state = np.r_[m, log_ratio]
            for _ in range(substeps):
                state = rk4_step(ode, state, step)
            m, log_ratio = state[0], state[1:]
        if gamma == 1:
            _, lam = rates(k, gamma, tau, m0)
            logs = -2 * lam * t
        else:
            d = np.sqrt(k / (k - 1)) * m ** (1 - gamma) / tau
            logs = np.array([
                2 * gamma * np.log(m0 / m) + 2 * (np.logaddexp(d0, np.log(k - 1)) - np.logaddexp(d, np.log(k - 1))),
                2 * gamma / (1 - gamma) * np.log(m0 / m) - 2 * (d - d0) / (k * (1 - gamma) ** 2),
                2 * gamma / (1 - gamma) * np.log(m0 / m)])
        checks.append(np.max(np.abs(logs - log_ratio)))
        blocks = initial_blocks * np.exp(logs)
        separation = np.sqrt(2 * k / (k - 1)) * m
        out.append(np.r_[separation, blocks.sum() / separation ** 2, blocks, m])
    return np.asarray(out), float(max(checks))


def general_moment_prediction(mu0, cov0, gamma, tau, times, step=.05):
    """Local full-matrix ODE, without a simplex ansatz or fitted parameters."""
    k = mu0.shape[-1]
    mu, cov = mu0.copy(), cov0.copy()
    state = np.concatenate([mu.reshape(-1), cov.reshape(-1)])
    def ode(x):
        centers = x[:k*k].reshape(k, k)
        sigma = x[k*k:].reshape(k, k, k)
        h = hessian(centers, np.arange(k), gamma, tau)
        return np.r_[field(centers, np.arange(k), gamma, tau).reshape(-1),
                     (-mm(h, sigma) - mm(sigma, h)).reshape(-1)]
    u, _, pr, pt, p0 = axes(k)
    output = []
    substeps = int(round((times[1] - times[0]) / step))
    for idx, _ in enumerate(times):
        if idx:
            for _ in range(substeps):
                state = rk4_step(ode, state, step)
        mu = state[:k*k].reshape(k, k)
        cov = state[k*k:].reshape(k, k, k)
        block = np.stack([np.einsum('cij,cji->c', pr, cov),
                          np.einsum('cij,cji->c', pt, cov),
                          np.einsum('ij,cji->c', p0, cov)], axis=1)
        i, j = np.triu_indices(k, 1)
        d2 = np.sum((mu[i] - mu[j])**2, axis=1)
        cdnv = np.mean((block.sum(1)[i] + block.sum(1)[j]) / (2 * d2))
        m = np.sqrt(np.mean(np.sum((mu - mu.mean(0))**2, axis=1)))
        output.append(np.r_[np.sqrt(d2.mean()), cdnv, block.mean(0), m])
    return np.asarray(output)
